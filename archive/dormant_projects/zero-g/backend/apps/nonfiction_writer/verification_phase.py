"""
Verification Phase for Nonfiction Writing

Fact-checking and claim verification using HoloLoom's agentic VERIFY mode.
Ensures all claims are supported by sources and identifies unsupported statements.

Verification Pipeline:
1. Extract claims from draft
2. Check each claim against sources
3. Identify contradictions and unsupported claims
4. Score confidence for each claim
5. Generate verification report

Integrates with:
- AgenticOrchestrator (VERIFY mode) for automated fact-checking
- ResearchCorpus for source verification
- CitationManager for tracking claim-source links
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

# Would integrate with actual HoloLoom in production
try:
    from HoloLoom.agentic import AgenticOrchestrator, ReasoningMode
    from HoloLoom.documentation.types import Query
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HOLOLOOM_AVAILABLE = False


from .draft_phase import Draft, Paragraph
from .research_phase import ResearchCorpus


class ClaimStatus(Enum):
    """Status of claim verification"""
    VERIFIED = "verified"              # Supported by sources
    UNSUPPORTED = "unsupported"        # No source support found
    CONTRADICTED = "contradicted"      # Contradicts sources
    UNCERTAIN = "uncertain"            # Ambiguous or conflicting sources


@dataclass
class Claim:
    """
    An extracted claim from the draft

    Attributes:
        id: Unique claim identifier
        text: Claim text
        paragraph_id: Source paragraph
        status: Verification status
        supporting_sources: Sources that support claim
        contradicting_sources: Sources that contradict claim
        confidence: Verification confidence (0.0-1.0)
        notes: Verification notes
    """
    id: str
    text: str
    paragraph_id: str
    status: ClaimStatus = ClaimStatus.UNCERTAIN
    supporting_sources: List[str] = field(default_factory=list)
    contradicting_sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""


@dataclass
class VerificationReport:
    """
    Complete verification report

    Attributes:
        draft: Draft that was verified
        claims: All extracted claims
        verified_count: Number of verified claims
        unsupported_count: Number of unsupported claims
        contradicted_count: Number of contradicted claims
        overall_confidence: Overall verification confidence
        issues: List of issues found
        recommendations: Recommendations for improvement
    """
    draft: Draft
    claims: List[Claim] = field(default_factory=list)
    verified_count: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0
    overall_confidence: float = 0.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def get_verification_rate(self) -> float:
        """Get percentage of verified claims"""
        if not self.claims:
            return 0.0
        return self.verified_count / len(self.claims)

    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = []
        lines.append("Verification Report")
        lines.append("=" * 50)
        lines.append(f"\nTotal claims: {len(self.claims)}")
        lines.append(f"Verified: {self.verified_count} ({self.get_verification_rate():.1%})")
        lines.append(f"Unsupported: {self.unsupported_count}")
        lines.append(f"Contradicted: {self.contradicted_count}")
        lines.append(f"Overall confidence: {self.overall_confidence:.2f}\n")

        if self.issues:
            lines.append("\nIssues Found:")
            for issue in self.issues:
                lines.append(f"  ⚠ {issue}")

        if self.recommendations:
            lines.append("\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  → {rec}")

        return "\n".join(lines)


class VerificationPhase:
    """
    Fact-checking and claim verification

    Uses agentic reasoning (VERIFY mode) to check all claims in draft
    against research sources and identify unsupported or contradicted claims.

    Usage:
        verifier = VerificationPhase(corpus)

        # Verify entire draft
        report = await verifier.verify(draft)

        # Check specific claim
        claim_status = await verifier.verify_claim(
            "Climate change causes sea level rise",
            sources
        )

        # Print report
        print(report.get_summary())
    """

    def __init__(self, corpus: ResearchCorpus):
        """
        Initialize verification phase

        Args:
            corpus: Research corpus for fact-checking
        """
        self.corpus = corpus
        self._claim_counter = 0

        # Initialize orchestrator if available
        self.orchestrator = None
        if HOLOLOOM_AVAILABLE:
            # Would initialize with proper config in production
            pass

    async def verify(self, draft: Draft) -> VerificationReport:
        """
        Verify all claims in draft

        Args:
            draft: Draft to verify

        Returns:
            VerificationReport with all findings
        """
        # Extract claims from draft
        claims = await self._extract_claims(draft)

        # Verify each claim
        for claim in claims:
            if HOLOLOOM_AVAILABLE and self.orchestrator:
                status, confidence = await self._verify_claim_with_agentic(claim)
            else:
                status, confidence = await self._verify_claim_simple(claim)

            claim.status = status
            claim.confidence = confidence

        # Generate report
        report = VerificationReport(draft=draft, claims=claims)

        # Count statuses
        report.verified_count = sum(1 for c in claims if c.status == ClaimStatus.VERIFIED)
        report.unsupported_count = sum(1 for c in claims if c.status == ClaimStatus.UNSUPPORTED)
        report.contradicted_count = sum(1 for c in claims if c.status == ClaimStatus.CONTRADICTED)

        # Calculate overall confidence
        if claims:
            report.overall_confidence = sum(c.confidence for c in claims) / len(claims)

        # Generate issues and recommendations
        report.issues, report.recommendations = self._analyze_verification(claims)

        return report

    async def verify_claim(self,
                          claim_text: str,
                          sources: Optional[List] = None) -> tuple[ClaimStatus, float]:
        """
        Verify a single claim

        Args:
            claim_text: Claim to verify
            sources: Sources to check against (None = use all corpus)

        Returns:
            (status, confidence) tuple
        """
        if sources is None:
            sources = self.corpus.sources

        claim = Claim(
            id=self._generate_claim_id(),
            text=claim_text,
            paragraph_id="",
        )

        if HOLOLOOM_AVAILABLE and self.orchestrator:
            return await self._verify_claim_with_agentic(claim)
        else:
            return await self._verify_claim_simple(claim)

    # ========================================================================
    # Private helper methods
    # ========================================================================

    def _generate_claim_id(self) -> str:
        """Generate unique claim ID"""
        self._claim_counter += 1
        return f"claim_{self._claim_counter:04d}"

    async def _extract_claims(self, draft: Draft) -> List[Claim]:
        """Extract claims from draft paragraphs"""
        claims = []

        for para in draft.paragraphs:
            # Split into sentences (simple heuristic)
            sentences = para.content.split('.')

            for sentence in sentences:
                sentence = sentence.strip()

                # Filter out non-claims (too short, questions, etc.)
                if len(sentence.split()) < 5:
                    continue
                if sentence.endswith('?'):
                    continue
                if sentence.startswith(('However', 'Moreover', 'Furthermore')):
                    # Remove transition words
                    sentence = sentence.split(',', 1)[-1].strip()

                # Create claim
                claim = Claim(
                    id=self._generate_claim_id(),
                    text=sentence,
                    paragraph_id=para.id,
                )

                claims.append(claim)

        return claims

    async def _verify_claim_with_agentic(self, claim: Claim) -> tuple[ClaimStatus, float]:
        """Verify claim using agentic VERIFY mode"""
        # Would use actual agentic orchestrator in full integration
        # query = Query(text=f"Verify this claim: {claim.text}")
        # result = await self.orchestrator.reason(query, mode=ReasoningMode.VERIFY)

        # Check result.verification for status and confidence

        # For MVP, fall back to simple
        return await self._verify_claim_simple(claim)

    async def _verify_claim_simple(self, claim: Claim) -> tuple[ClaimStatus, float]:
        """Simple claim verification"""
        # Check if claim keywords appear in sources
        claim_words = set(claim.text.lower().split())
        claim_words = {w for w in claim_words if len(w) > 4}  # Filter short words

        supporting_sources = []
        contradicting_sources = []

        for source in self.corpus.sources:
            source_text = source.summary.lower()

            # Count matching keywords
            matches = sum(1 for word in claim_words if word in source_text)
            match_rate = matches / max(1, len(claim_words))

            if match_rate > 0.4:
                supporting_sources.append(source.id)
                claim.supporting_sources.append(source.id)

            # Check for contradictory terms
            contradictory_pairs = [
                ('increase', 'decrease'),
                ('positive', 'negative'),
                ('support', 'contradict'),
            ]

            for word1, word2 in contradictory_pairs:
                if word1 in claim.text.lower() and word2 in source_text:
                    contradicting_sources.append(source.id)
                    claim.contradicting_sources.append(source.id)

        # Determine status
        if contradicting_sources:
            status = ClaimStatus.CONTRADICTED
            confidence = 0.3
        elif supporting_sources:
            status = ClaimStatus.VERIFIED
            confidence = min(0.9, len(supporting_sources) / 3)  # Max 0.9 with 3+ sources
        elif len(claim_words) < 3:
            status = ClaimStatus.UNCERTAIN
            confidence = 0.5
        else:
            status = ClaimStatus.UNSUPPORTED
            confidence = 0.2

        return status, confidence

    def _analyze_verification(self, claims: List[Claim]) -> tuple[List[str], List[str]]:
        """Analyze claims and generate issues/recommendations"""
        issues = []
        recommendations = []

        # Check for high percentage of unsupported claims
        unsupported_rate = sum(1 for c in claims if c.status == ClaimStatus.UNSUPPORTED) / max(1, len(claims))

        if unsupported_rate > 0.3:
            issues.append(f"{unsupported_rate:.1%} of claims lack source support")
            recommendations.append("Add citations for unsupported claims or remove them")

        # Check for contradicted claims
        contradicted = [c for c in claims if c.status == ClaimStatus.CONTRADICTED]
        if contradicted:
            issues.append(f"{len(contradicted)} claims contradict research sources")
            recommendations.append("Review and revise contradicted claims")

        # Check for low-confidence verified claims
        low_confidence = [c for c in claims if c.status == ClaimStatus.VERIFIED and c.confidence < 0.5]
        if low_confidence:
            recommendations.append(f"Strengthen {len(low_confidence)} low-confidence claims with additional sources")

        # Check overall confidence
        if claims:
            avg_confidence = sum(c.confidence for c in claims) / len(claims)
            if avg_confidence < 0.6:
                issues.append(f"Overall confidence is low ({avg_confidence:.2f})")
                recommendations.append("Add more high-quality sources to support claims")

        return issues, recommendations


# Factory function
def create_verification_phase(corpus: ResearchCorpus) -> VerificationPhase:
    """Create a new verification phase"""
    return VerificationPhase(corpus)
