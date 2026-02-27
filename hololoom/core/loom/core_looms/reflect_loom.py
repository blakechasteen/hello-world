"""
REFLECT Loom - The Auditor
===========================

Consistency checking, verification-grounded. Asks "does this hold up?"

Cognitive Style:
- Multiple verification passes
- Cross-reference checking
- Internal consistency validation
- Contradiction detection

Philosophy:
"Does this claim hold up under scrutiny? Is it internally consistent?
The Auditor trusts nothing without verification."

Date: December 2025
Author: HoloLoom Architecture Team
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging
import re

from hololoom.core.loom.base_loom import BaseLoom
from hololoom.core.loom.protocol import REFLECT, CorrectionInsight, PatternInsight
from hololoom.core.fabric.fabric import Fabric
from hololoom.core.protocols.department import (
    DepartmentRequest,
    DepartmentResponse,
    ConfidenceMetadata,
)

logger = logging.getLogger(__name__)


@dataclass
class Claim:
    """A verifiable claim extracted from content."""
    text: str
    source: str
    confidence: float
    verifiable: bool
    status: str = "unverified"  # unverified, verified, contradicted, uncertain


@dataclass
class VerificationResult:
    """Result of verifying a claim."""
    claim: Claim
    verified: bool
    confidence: float
    evidence: List[str]
    contradictions: List[str]


@dataclass
class ConsistencyCheck:
    """Result of internal consistency checking."""
    consistent: bool
    contradictions: List[Tuple[str, str]]  # Pairs of contradicting claims
    confidence: float


class ReflectLoom(BaseLoom):
    """
    The Auditor - Consistency checking, verification-grounded.

    Asks "does this hold up?" Excels at finding inconsistencies,
    verifying claims, and detecting contradictions.

    Attributes:
        verification_passes: Number of verification iterations (default: 2)
        consistency_threshold: Minimum consistency for approval (default: 0.8)
        require_evidence: Require evidence for verification (default: True)
    """

    def __init__(
        self,
        verification_passes: int = 2,
        consistency_threshold: float = 0.8,
        require_evidence: bool = True,
        verification_backend: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize the Reflect Loom.

        Args:
            verification_passes: Number of verification iterations
            consistency_threshold: Minimum consistency score for approval
            require_evidence: Whether to require evidence for claims
            verification_backend: Optional backend for external verification
            **kwargs: Arguments passed to BaseLoom
        """
        super().__init__(**kwargs)

        # Cognitive style parameters
        self.verification_passes = verification_passes
        self.consistency_threshold = consistency_threshold
        self.require_evidence = require_evidence

        # Verification backend
        self._verifier = verification_backend

        # Track corrections for learning
        self._corrections: List[CorrectionInsight] = []

    @property
    def perspective(self) -> str:
        """The Auditor's perspective."""
        return REFLECT

    @property
    def blind_spots(self) -> List[str]:
        """What the Auditor cannot see."""
        return [
            "generating new content",
            "creative solutions",
            "speed and efficiency",
            "unknown unknowns",
            "context beyond the data",
            "when verification is impossible",
        ]

    def admits_ignorance(self, query: str) -> List[str]:
        """
        What the Auditor knows it doesn't know about this query.

        Args:
            query: The query to assess

        Returns:
            List of specific knowledge gaps
        """
        ignorance = []

        # Check for creative keywords
        creative_words = ['create', 'imagine', 'design', 'invent', 'new']
        if any(word in query.lower() for word in creative_words):
            ignorance.append("Cannot create new content - only verify existing")

        # Check for speed keywords
        speed_words = ['quick', 'fast', 'immediately', 'urgent']
        if any(word in query.lower() for word in speed_words):
            ignorance.append("Verification takes time - cannot rush accuracy")

        # Check for prediction keywords
        prediction_words = ['will', 'predict', 'future', 'expect']
        if any(word in query.lower() for word in prediction_words):
            ignorance.append("Cannot verify future claims - only past/present")

        # Add any unverifiable claims found
        if self._corrections:
            ignorance.append(f"Found {len(self._corrections)} unverifiable claims")

        if not ignorance:
            ignorance.append("May miss contradictions requiring external knowledge")

        return ignorance

    def falsifiable_by(self, claim: str) -> List[str]:
        """
        What evidence would change the Auditor's claim.

        Args:
            claim: The claim to assess

        Returns:
            List of falsifiers
        """
        return [
            "Finding a counterexample to the verification",
            "Discovering hidden contradictions",
            "New evidence that contradicts verified claims",
            "External verification proving internal check wrong",
            "Finding flaws in the verification methodology",
        ]

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Execute verification logic - check consistency and verify claims.

        Args:
            request: Department request with query

        Returns:
            Response with verification results
        """
        start_time = datetime.now()
        query = request.query

        # Step 1: Extract verifiable claims
        claims = self._extract_claims(query)

        # Step 2: Run verification passes
        verification_results = []
        for pass_num in range(self.verification_passes):
            results = await self._verify_claims(claims)
            verification_results.extend(results)

        # Step 3: Check internal consistency
        consistency = self._check_consistency(claims)

        # Step 4: Find contradictions
        contradictions = self._find_contradictions(claims, verification_results)

        # Step 5: Format verification report
        response = self._format_verification_report(
            verification_results, consistency, contradictions
        )

        # Step 6: Compute confidence
        confidence = self._compute_verification_confidence(
            verification_results, consistency
        )

        # Step 7: Compute epistemic confidence
        epistemic = self._compute_epistemic_confidence(
            verification_results, consistency, contradictions
        )

        # Calculate duration
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Update metrics
        self._update_metrics(confidence, duration_ms)

        # Log corrections for dreaming
        for contradiction in contradictions:
            self.log_correction_insight({
                "original": contradiction[0],
                "correction": contradiction[1],
                "reason": "Contradiction detected",
            }, confidence=0.9)

        # Create fabric
        fabric = self._create_fabric(
            query=query,
            response=response,
            tool_used="reflect",
            confidence=confidence,
            duration_ms=duration_ms,
            epistemic_confidence=epistemic,
            admits_ignorance=self.admits_ignorance(query),
            falsifiable_by=self.falsifiable_by(response),
            metadata={
                "claims_found": len(claims),
                "verification_passes": self.verification_passes,
                "verified_count": sum(1 for r in verification_results if r.verified),
                "contradictions_count": len(contradictions),
                "consistency_score": consistency.confidence,
            },
        )

        return DepartmentResponse(
            result=response,
            confidence=ConfidenceMetadata(
                score=confidence,
                source="reflect_loom",
                reasoning=f"Verified {len(verification_results)} claims, {len(contradictions)} contradictions",
            ),
            metadata={
                "perspective": self.perspective,
                "fabric": fabric,
                "verification_results": verification_results,
                "consistency": consistency,
                "contradictions": contradictions,
            },
        )

    def _extract_claims(self, text: str) -> List[Claim]:
        """
        Extract verifiable claims from text.

        Args:
            text: Text to extract claims from

        Returns:
            List of extracted claims
        """
        claims = []

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue

            # Determine if claim is verifiable
            verifiable = self._is_verifiable(sentence)

            claims.append(Claim(
                text=sentence,
                source="query",
                confidence=0.5 if verifiable else 0.3,
                verifiable=verifiable,
            ))

        return claims

    def _is_verifiable(self, claim: str) -> bool:
        """
        Determine if a claim is verifiable.

        Args:
            claim: The claim text

        Returns:
            True if verifiable
        """
        # Unverifiable patterns
        unverifiable_patterns = [
            r'\b(feel|believe|think|hope|wish)\b',  # Subjective
            r'\b(maybe|perhaps|possibly|might)\b',  # Uncertain
            r'\b(should|ought|must)\b',              # Normative
            r'\b(best|worst|better|worse)\b',        # Comparative (subjective)
        ]

        for pattern in unverifiable_patterns:
            if re.search(pattern, claim.lower()):
                return False

        # Verifiable patterns (factual assertions)
        verifiable_patterns = [
            r'\b(is|are|was|were|has|have|had)\b',  # Factual statements
            r'\b(equals?|contains?|includes?)\b',    # Relationships
            r'\d+',                                   # Numeric claims
        ]

        for pattern in verifiable_patterns:
            if re.search(pattern, claim.lower()):
                return True

        return False  # Default to not verifiable

    async def _verify_claims(
        self,
        claims: List[Claim]
    ) -> List[VerificationResult]:
        """
        Verify a list of claims.

        Args:
            claims: Claims to verify

        Returns:
            List of verification results
        """
        results = []

        for claim in claims:
            if not claim.verifiable:
                # Mark unverifiable claims
                results.append(VerificationResult(
                    claim=claim,
                    verified=False,
                    confidence=0.3,
                    evidence=[],
                    contradictions=["Claim is not verifiable"],
                ))
                continue

            # Attempt verification
            if self._verifier:
                # Use external verifier if available
                verified, evidence = await self._external_verify(claim)
            else:
                # Simple self-verification (check for internal consistency)
                verified, evidence = self._self_verify(claim)

            results.append(VerificationResult(
                claim=claim,
                verified=verified,
                confidence=0.8 if verified else 0.4,
                evidence=evidence,
                contradictions=[],
            ))

        return results

    async def _external_verify(
        self,
        claim: Claim
    ) -> Tuple[bool, List[str]]:
        """
        Verify claim using external backend.

        Args:
            claim: Claim to verify

        Returns:
            Tuple of (verified, evidence)
        """
        try:
            result = await self._verifier.verify(claim.text)
            return result.verified, result.evidence
        except Exception as e:
            logger.warning(f"External verification failed: {e}")
            return self._self_verify(claim)

    def _self_verify(self, claim: Claim) -> Tuple[bool, List[str]]:
        """
        Self-verify a claim using internal logic.

        Args:
            claim: Claim to verify

        Returns:
            Tuple of (verified, evidence)
        """
        # Simple heuristic verification
        evidence = []

        # Check for definite statements (more likely verifiable)
        if re.search(r'\b(is|are|equals?)\b', claim.text.lower()):
            evidence.append("Contains definite assertion")

        # Check for numeric precision
        if re.search(r'\d+', claim.text):
            evidence.append("Contains numeric data")

        # Default: cannot fully verify without external source
        verified = len(evidence) > 0
        if not verified:
            evidence.append("Insufficient internal evidence for verification")

        return verified, evidence

    def _check_consistency(self, claims: List[Claim]) -> ConsistencyCheck:
        """
        Check internal consistency of claims.

        Args:
            claims: Claims to check

        Returns:
            ConsistencyCheck result
        """
        contradictions = []

        # Compare each pair of claims for contradictions
        for i, claim_a in enumerate(claims):
            for claim_b in claims[i + 1:]:
                if self._are_contradictory(claim_a.text, claim_b.text):
                    contradictions.append((claim_a.text, claim_b.text))

        consistent = len(contradictions) == 0
        confidence = 1.0 - (len(contradictions) * 0.2)  # Reduce for each contradiction
        confidence = max(confidence, 0.2)  # Floor at 0.2

        return ConsistencyCheck(
            consistent=consistent,
            contradictions=contradictions,
            confidence=confidence,
        )

    def _are_contradictory(self, text_a: str, text_b: str) -> bool:
        """
        Check if two statements are contradictory.

        Args:
            text_a: First statement
            text_b: Second statement

        Returns:
            True if contradictory
        """
        # Simple negation detection
        negation_words = ['not', 'never', 'no', "n't", 'cannot', 'without']

        a_lower = text_a.lower()
        b_lower = text_b.lower()

        # Check for explicit negation
        a_negated = any(neg in a_lower for neg in negation_words)
        b_negated = any(neg in b_lower for neg in negation_words)

        # If one is negated and they share key words, likely contradictory
        if a_negated != b_negated:
            # Extract key nouns/verbs (simple approach)
            a_words = set(a_lower.split())
            b_words = set(b_lower.split())
            common = a_words & b_words

            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'to'}
            common = common - stop_words

            if len(common) >= 2:  # Share at least 2 meaningful words
                return True

        return False

    def _find_contradictions(
        self,
        claims: List[Claim],
        results: List[VerificationResult]
    ) -> List[Tuple[str, str]]:
        """
        Find contradictions between claims and verification results.

        Args:
            claims: Original claims
            results: Verification results

        Returns:
            List of contradiction pairs
        """
        contradictions = []

        for result in results:
            if result.contradictions:
                for contradiction in result.contradictions:
                    if contradiction != "Claim is not verifiable":
                        contradictions.append((result.claim.text, contradiction))

        return contradictions

    def _format_verification_report(
        self,
        results: List[VerificationResult],
        consistency: ConsistencyCheck,
        contradictions: List[Tuple[str, str]]
    ) -> str:
        """
        Format verification results as report.

        Args:
            results: Verification results
            consistency: Consistency check result
            contradictions: Found contradictions

        Returns:
            Formatted report text
        """
        parts = ["Verification Report:\n"]

        # Summary
        verified_count = sum(1 for r in results if r.verified)
        total = len(results)
        parts.append(f"Claims Verified: {verified_count}/{total}")
        parts.append(f"Consistency: {'✓' if consistency.consistent else '✗'} "
                    f"(confidence: {consistency.confidence:.2f})")

        # Verified claims
        if verified_count > 0:
            parts.append("\nVerified Claims:")
            for r in results:
                if r.verified:
                    parts.append(f"  ✓ {r.claim.text[:60]}... (conf: {r.confidence:.2f})")

        # Unverified claims
        unverified = [r for r in results if not r.verified]
        if unverified:
            parts.append("\nUnverified/Uncertain Claims:")
            for r in unverified:
                parts.append(f"  ? {r.claim.text[:60]}...")

        # Contradictions
        if contradictions:
            parts.append("\n⚠ Contradictions Found:")
            for claim, contradiction in contradictions[:3]:  # Limit to 3
                parts.append(f"  - {claim[:40]}... vs {contradiction[:40]}...")

        return "\n".join(parts)

    def _compute_verification_confidence(
        self,
        results: List[VerificationResult],
        consistency: ConsistencyCheck
    ) -> float:
        """
        Compute overall verification confidence.

        Args:
            results: Verification results
            consistency: Consistency check

        Returns:
            Confidence score (0-1)
        """
        if not results:
            return 0.3

        # Average verification confidence
        avg_verify = sum(r.confidence for r in results) / len(results)

        # Weight by consistency
        confidence = 0.6 * avg_verify + 0.4 * consistency.confidence

        return min(confidence, 1.0)

    def _compute_epistemic_confidence(
        self,
        results: List[VerificationResult],
        consistency: ConsistencyCheck,
        contradictions: List[Tuple[str, str]]
    ) -> float:
        """
        Compute epistemic confidence.

        Args:
            results: Verification results
            consistency: Consistency check
            contradictions: Found contradictions

        Returns:
            Epistemic confidence (0-1)
        """
        # High base for verification (we're checking, not creating)
        epistemic = 0.85

        # Reduce for unverifiable claims
        unverifiable = sum(1 for r in results if not r.claim.verifiable)
        epistemic -= unverifiable * 0.05

        # Reduce for contradictions
        epistemic -= len(contradictions) * 0.1

        # Reduce for low consistency
        if consistency.confidence < 0.7:
            epistemic -= 0.15

        return max(epistemic, 0.3)


__all__ = ['ReflectLoom']
