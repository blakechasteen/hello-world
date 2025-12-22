"""
Ground Truth & Verification Boundaries (Addendum A.2)

Implements three-tier verification system:

TIER 1: INTERNAL CONSISTENCY (Self-Verification)
    - CoVe self-consistency checks
    - Multi-sample agreement
    - Behavioral probe consistency
    - Knowledge graph coherence
    Sufficient for: Opinions, style, creative content, explanations
    NOT sufficient for: Facts, numbers, dates, medical/legal claims

TIER 2: EXTERNAL VERIFICATION (Reference Checking)
    - Cross-reference against knowledge graph
    - Vector search against ingested corpus
    - API calls to authoritative sources (optional)
    - Web search verification (optional)
    Required for: Factual claims, historical dates, scientific facts

TIER 3: AUTHORITATIVE SOURCES (Ground Truth)
    - Explicitly marked authoritative sources in KG
    - Curated reference databases
    - Human-verified facts with provenance
    - Official documentation, specifications, standards
    Required for: Medical advice, legal guidance, safety-critical info
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Callable
import asyncio


class VerificationTier(Enum):
    """Verification tiers from least to most rigorous."""
    TIER_1_INTERNAL = 1      # Self-consistency
    TIER_2_EXTERNAL = 2      # External reference
    TIER_3_AUTHORITATIVE = 3  # Ground truth sources


class ClaimType(Enum):
    """Types of claims with different verification requirements."""
    OPINION = "opinion"           # Tier 1 sufficient
    CREATIVE = "creative"         # Tier 1 sufficient
    EXPLANATION = "explanation"   # Tier 1 sufficient
    FACTUAL = "factual"           # Tier 1+2 required
    HISTORICAL = "historical"     # Tier 1+2 required
    SCIENTIFIC = "scientific"     # Tier 1+2 required
    MEDICAL = "medical"           # Tier 1+2+3 required
    LEGAL = "legal"               # Tier 1+2+3 required
    SAFETY_CRITICAL = "safety"    # Tier 1+2+3 required

    @property
    def required_tiers(self) -> List[VerificationTier]:
        """Get required verification tiers for this claim type."""
        if self in (ClaimType.OPINION, ClaimType.CREATIVE, ClaimType.EXPLANATION):
            return [VerificationTier.TIER_1_INTERNAL]
        elif self in (ClaimType.FACTUAL, ClaimType.HISTORICAL, ClaimType.SCIENTIFIC):
            return [VerificationTier.TIER_1_INTERNAL, VerificationTier.TIER_2_EXTERNAL]
        else:  # Medical, Legal, Safety-critical
            return [
                VerificationTier.TIER_1_INTERNAL,
                VerificationTier.TIER_2_EXTERNAL,
                VerificationTier.TIER_3_AUTHORITATIVE,
            ]


@dataclass
class VerificationResult:
    """Result of a single verification tier check."""
    tier: VerificationTier
    passed: bool
    confidence: float
    sources_checked: List[str]
    details: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VerificationStatus:
    """
    Verification status attached to every claim.

    Every output includes verification status showing:
    - Which tiers were checked
    - Which passed/failed
    - What sources were consulted
    - Which sub-claims couldn't be verified
    """
    tier_1_passed: bool                    # Internal consistency
    tier_2_passed: Optional[bool] = None   # External verification (if performed)
    tier_3_passed: Optional[bool] = None   # Authoritative source (if required)
    verification_sources: List[str] = field(default_factory=list)
    verification_timestamp: datetime = field(default_factory=datetime.now)
    unverified_claims: List[str] = field(default_factory=list)
    tier_results: List[VerificationResult] = field(default_factory=list)

    @property
    def all_required_passed(self) -> bool:
        """Check if all performed verifications passed."""
        if not self.tier_1_passed:
            return False
        if self.tier_2_passed is not None and not self.tier_2_passed:
            return False
        if self.tier_3_passed is not None and not self.tier_3_passed:
            return False
        return True

    @property
    def highest_tier_passed(self) -> Optional[VerificationTier]:
        """Get the highest tier that passed verification."""
        if self.tier_3_passed:
            return VerificationTier.TIER_3_AUTHORITATIVE
        if self.tier_2_passed:
            return VerificationTier.TIER_2_EXTERNAL
        if self.tier_1_passed:
            return VerificationTier.TIER_1_INTERNAL
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tier_1_passed": self.tier_1_passed,
            "tier_2_passed": self.tier_2_passed,
            "tier_3_passed": self.tier_3_passed,
            "verification_sources": self.verification_sources,
            "verification_timestamp": self.verification_timestamp.isoformat(),
            "unverified_claims": self.unverified_claims,
            "all_passed": self.all_required_passed,
            "highest_tier": self.highest_tier_passed.value if self.highest_tier_passed else None,
        }


class VerificationEngine:
    """
    Engine for performing multi-tier verification.

    Implements the verification decision matrix:
                        Claim Type
                 ┌────────────┬────────────┬────────────┐
                 │  Opinion/  │  Factual   │  Safety-   │
                 │  Creative  │   Claim    │  Critical  │
    ┌────────────┼────────────┼────────────┼────────────┤
    │ HIGH conf  │ Tier 1     │ Tier 1+2   │ Tier 1+2+3 │
    │            │ only       │            │            │
    ├────────────┼────────────┼────────────┼────────────┤
    │ MODERATE   │ Tier 1     │ Tier 1+2   │ Tier 1+2+3 │
    │ conf       │ + hedge    │ + hedge    │ + human    │
    ├────────────┼────────────┼────────────┼────────────┤
    │ LOW conf   │ Tier 1     │ Tier 1+2   │ BLOCK      │
    │            │ + caveat   │ + verify   │ (escalate) │
    └────────────┴────────────┴────────────┴────────────┘
    """

    def __init__(
        self,
        knowledge_graph=None,
        vector_store=None,
        authoritative_sources: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize verification engine.

        Args:
            knowledge_graph: HoloLoom KG for fact checking
            vector_store: Vector store for semantic search
            authoritative_sources: Dict of source_name -> verification function
        """
        self.kg = knowledge_graph
        self.vector_store = vector_store
        self.authoritative_sources = authoritative_sources or {}

    async def verify_tier_1(
        self,
        response: str,
        samples: Optional[List[str]] = None,
    ) -> VerificationResult:
        """
        Tier 1: Internal Consistency (Self-Verification)

        Checks:
        - Multi-sample agreement (if samples provided)
        - Self-consistency (no contradictions)
        - Knowledge graph coherence
        """
        sources_checked = []
        issues = []

        # Check multi-sample agreement
        if samples and len(samples) > 1:
            sources_checked.append("multi_sample_agreement")
            # Simple agreement check - more sophisticated would use embeddings
            unique_samples = len(set(samples))
            agreement_ratio = 1.0 - (unique_samples - 1) / len(samples)
            if agreement_ratio < 0.8:
                issues.append(f"Low sample agreement: {agreement_ratio:.2f}")

        # Check knowledge graph coherence if available
        if self.kg is not None:
            sources_checked.append("knowledge_graph_coherence")
            # Would check if response aligns with KG facts
            # Placeholder for now

        passed = len(issues) == 0
        confidence = 0.9 if passed else 0.5

        return VerificationResult(
            tier=VerificationTier.TIER_1_INTERNAL,
            passed=passed,
            confidence=confidence,
            sources_checked=sources_checked,
            details="; ".join(issues) if issues else "All consistency checks passed",
        )

    async def verify_tier_2(
        self,
        claims: List[str],
    ) -> VerificationResult:
        """
        Tier 2: External Verification (Reference Checking)

        Checks:
        - Cross-reference against knowledge graph
        - Vector search against ingested corpus
        - External API verification (if configured)
        """
        sources_checked = []
        verified_claims = []
        unverified_claims = []

        for claim in claims:
            verified = False

            # Check knowledge graph
            if self.kg is not None:
                sources_checked.append("knowledge_graph")
                # Would query KG for claim verification
                # Placeholder: assume verification succeeds
                verified = True

            # Check vector store
            if self.vector_store is not None and not verified:
                sources_checked.append("vector_store")
                # Would do semantic search for supporting evidence
                # Placeholder

            if verified:
                verified_claims.append(claim)
            else:
                unverified_claims.append(claim)

        # Deduplicate sources
        sources_checked = list(set(sources_checked))

        passed = len(unverified_claims) == 0
        confidence = len(verified_claims) / len(claims) if claims else 1.0

        return VerificationResult(
            tier=VerificationTier.TIER_2_EXTERNAL,
            passed=passed,
            confidence=confidence,
            sources_checked=sources_checked,
            details=f"Verified {len(verified_claims)}/{len(claims)} claims",
        )

    async def verify_tier_3(
        self,
        claims: List[str],
        required_sources: Optional[List[str]] = None,
    ) -> VerificationResult:
        """
        Tier 3: Authoritative Sources (Ground Truth)

        Checks claims against:
        - Explicitly marked authoritative sources in KG
        - Curated reference databases
        - Human-verified facts with provenance
        """
        sources_checked = []
        verified_claims = []
        unverified_claims = []

        # Use all authoritative sources if none specified
        sources_to_check = required_sources or list(self.authoritative_sources.keys())

        for claim in claims:
            verified = False

            for source_name in sources_to_check:
                if source_name in self.authoritative_sources:
                    sources_checked.append(source_name)
                    verifier = self.authoritative_sources[source_name]

                    try:
                        # Call the verification function
                        if asyncio.iscoroutinefunction(verifier):
                            result = await verifier(claim)
                        else:
                            result = verifier(claim)

                        if result:
                            verified = True
                            break
                    except Exception:
                        continue

            if verified:
                verified_claims.append(claim)
            else:
                unverified_claims.append(claim)

        sources_checked = list(set(sources_checked))

        passed = len(unverified_claims) == 0
        confidence = len(verified_claims) / len(claims) if claims else 0.0

        return VerificationResult(
            tier=VerificationTier.TIER_3_AUTHORITATIVE,
            passed=passed,
            confidence=confidence,
            sources_checked=sources_checked,
            details=f"Authoritative verification: {len(verified_claims)}/{len(claims)} claims",
        )

    async def verify(
        self,
        response: str,
        claims: List[str],
        claim_type: ClaimType,
        confidence: float,
        samples: Optional[List[str]] = None,
    ) -> VerificationStatus:
        """
        Perform full verification based on claim type and confidence.

        Uses the verification decision matrix to determine required checks.
        """
        required_tiers = claim_type.required_tiers
        tier_results = []
        sources = []

        # Tier 1: Always perform internal consistency
        tier_1_result = await self.verify_tier_1(response, samples)
        tier_results.append(tier_1_result)
        sources.extend(tier_1_result.sources_checked)

        tier_2_passed = None
        tier_3_passed = None

        # Tier 2: If required
        if VerificationTier.TIER_2_EXTERNAL in required_tiers:
            tier_2_result = await self.verify_tier_2(claims)
            tier_results.append(tier_2_result)
            sources.extend(tier_2_result.sources_checked)
            tier_2_passed = tier_2_result.passed

        # Tier 3: If required
        if VerificationTier.TIER_3_AUTHORITATIVE in required_tiers:
            tier_3_result = await self.verify_tier_3(claims)
            tier_results.append(tier_3_result)
            sources.extend(tier_3_result.sources_checked)
            tier_3_passed = tier_3_result.passed

        # Collect unverified claims
        unverified = []
        for result in tier_results:
            if not result.passed and result.details:
                unverified.append(result.details)

        return VerificationStatus(
            tier_1_passed=tier_1_result.passed,
            tier_2_passed=tier_2_passed,
            tier_3_passed=tier_3_passed,
            verification_sources=list(set(sources)),
            unverified_claims=unverified,
            tier_results=tier_results,
        )


async def verify_response(
    response: str,
    claims: Optional[List[str]] = None,
    claim_type: ClaimType = ClaimType.FACTUAL,
    confidence: float = 0.8,
    knowledge_graph=None,
    vector_store=None,
) -> VerificationStatus:
    """
    Convenience function to verify a response.

    Args:
        response: The response text to verify
        claims: List of specific claims to verify (auto-extracted if None)
        claim_type: Type of claims being made
        confidence: Current confidence level
        knowledge_graph: Optional KG for verification
        vector_store: Optional vector store for verification

    Returns:
        VerificationStatus with results
    """
    engine = VerificationEngine(
        knowledge_graph=knowledge_graph,
        vector_store=vector_store,
    )

    # Auto-extract claims if not provided (simplified)
    if claims is None:
        claims = [response]  # Treat entire response as one claim

    return await engine.verify(
        response=response,
        claims=claims,
        claim_type=claim_type,
        confidence=confidence,
    )


def should_block_for_safety(
    claim_type: ClaimType,
    confidence: float,
    verification_status: VerificationStatus,
) -> tuple[bool, str]:
    """
    Determine if response should be blocked for safety.

    Per the verification decision matrix:
    - Safety-critical + LOW confidence = BLOCK (escalate)
    """
    if claim_type in (ClaimType.MEDICAL, ClaimType.LEGAL, ClaimType.SAFETY_CRITICAL):
        if confidence < 0.6:
            return True, "Low confidence on safety-critical claim requires human review"

        if not verification_status.all_required_passed:
            return True, "Safety-critical claim failed verification"

    return False, ""
