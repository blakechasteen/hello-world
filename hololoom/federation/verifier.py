"""
Consensus Verification - Safety through agreement.

Multiple nodes must agree before a response is trusted.
Disagreement is information, not failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher

from .types import (
    Query,
    Response,
    Verification,
    VerificationLevel,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  VERIFICATION RESULT
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class VerificationScore:
    """Detailed verification scoring."""

    domain: float = 0.0          # Domain relevance
    sensibility: float = 0.0     # Makes sense
    temporal: float = 0.0        # Temporally accurate
    argument: float = 0.0        # Logical consistency
    reference: float = 0.0       # Source quality

    @property
    def total(self) -> float:
        """Weighted total score."""
        weights = {
            "domain": 0.20,
            "sensibility": 0.25,
            "temporal": 0.15,
            "argument": 0.25,
            "reference": 0.15,
        }
        return (
            self.domain * weights["domain"]
            + self.sensibility * weights["sensibility"]
            + self.temporal * weights["temporal"]
            + self.argument * weights["argument"]
            + self.reference * weights["reference"]
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "domain": self.domain,
            "sensibility": self.sensibility,
            "temporal": self.temporal,
            "argument": self.argument,
            "reference": self.reference,
            "total": self.total,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  QUORUM REQUIREMENTS
# ═══════════════════════════════════════════════════════════════════════════


def get_quorum(level: VerificationLevel) -> int:
    """Get required quorum for verification level."""
    return {
        VerificationLevel.NONE: 0,
        VerificationLevel.LIGHT: 2,
        VerificationLevel.STANDARD: 3,
        VerificationLevel.DEEP: 5,
        VerificationLevel.CRITICAL: 7,
    }[level]


def get_agreement_threshold(level: VerificationLevel) -> float:
    """Get required agreement threshold (0.0-1.0)."""
    return {
        VerificationLevel.NONE: 0.0,
        VerificationLevel.LIGHT: 0.50,
        VerificationLevel.STANDARD: 0.66,
        VerificationLevel.DEEP: 0.75,
        VerificationLevel.CRITICAL: 0.85,
    }[level]


# ═══════════════════════════════════════════════════════════════════════════
#  CONSENSUS VERIFIER
# ═══════════════════════════════════════════════════════════════════════════


class ConsensusVerifier:
    """
    Multi-node consensus verification.

    Collects responses from multiple nodes and determines:
    1. Whether they agree sufficiently
    2. What the consensus response should be
    3. Which nodes dissented and why

    Usage:
        verifier = ConsensusVerifier()

        verification = await verifier.verify(
            query=query,
            responses=responses,
            level=VerificationLevel.STANDARD,
        )

        if verification.verified:
            print(f"Consensus reached: {verification.consensus_response}")
        else:
            print(f"Disagreement: {verification.dissenting}")
    """

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.8,
        min_confidence: float = 0.3,
    ):
        self._similarity_threshold = similarity_threshold
        self._min_confidence = min_confidence

    # ───────────────────────────────────────────────────────────────────────
    #  MAIN VERIFICATION
    # ───────────────────────────────────────────────────────────────────────

    async def verify(
        self,
        query: Query,
        responses: list[Response],
        *,
        level: VerificationLevel,
        quorum: int | None = None,
    ) -> Verification:
        """
        Verify responses through consensus.

        Args:
            query: Original query
            responses: Candidate responses from nodes
            level: Verification thoroughness
            quorum: Override quorum (default: from level)

        Returns:
            Verification result
        """
        if level == VerificationLevel.NONE:
            return self._skip_verification(query, responses)

        quorum = quorum or get_quorum(level)
        threshold = get_agreement_threshold(level)

        # Filter low-confidence responses
        valid = [r for r in responses if r.confidence >= self._min_confidence]

        if len(valid) < quorum:
            return self._insufficient_quorum(query, valid, quorum)

        # Cluster similar responses
        clusters = self._cluster_responses(valid)

        # Find largest agreeing cluster
        largest = max(clusters, key=len) if clusters else []
        agreement_ratio = len(largest) / len(valid)

        if agreement_ratio >= threshold:
            # Consensus reached
            consensus = self._merge_responses(largest)
            verifiers = frozenset(r.responder for r in largest)
            dissenting = frozenset(
                r.responder for r in valid if r.responder not in verifiers
            )

            return Verification(
                request_id=query.request_id,
                verified=True,
                confidence=agreement_ratio,
                consensus_response=consensus,
                verifiers=verifiers,
                dissenting=dissenting,
                scores=self._aggregate_scores(largest),
            )
        else:
            # No consensus
            return self._no_consensus(query, valid, clusters)

    def _skip_verification(
        self,
        query: Query,
        responses: list[Response],
    ) -> Verification:
        """Skip verification (NONE level)."""
        if not responses:
            return Verification(
                request_id=query.request_id,
                verified=False,
                confidence=0.0,
                consensus_response="",
                verifiers=frozenset(),
            )

        # Just use highest confidence response
        best = max(responses, key=lambda r: r.confidence)
        return Verification(
            request_id=query.request_id,
            verified=True,
            confidence=best.confidence,
            consensus_response=best.text,
            verifiers=frozenset([best.responder]),
        )

    def _insufficient_quorum(
        self,
        query: Query,
        responses: list[Response],
        quorum: int,
    ) -> Verification:
        """Handle insufficient quorum."""
        logger.warning(
            f"Insufficient quorum: {len(responses)}/{quorum} for query {query.request_id[:8]}"
        )

        return Verification(
            request_id=query.request_id,
            verified=False,
            confidence=0.0,
            consensus_response="",
            verifiers=frozenset(r.responder for r in responses),
            scores={"error": f"insufficient_quorum:{len(responses)}/{quorum}"},
        )

    def _no_consensus(
        self,
        query: Query,
        responses: list[Response],
        clusters: list[list[Response]],
    ) -> Verification:
        """Handle no consensus."""
        logger.warning(
            f"No consensus for query {query.request_id[:8]}: {len(clusters)} clusters"
        )

        # Return best cluster anyway, but mark as unverified
        largest = max(clusters, key=len) if clusters else []

        return Verification(
            request_id=query.request_id,
            verified=False,
            confidence=len(largest) / len(responses) if responses else 0.0,
            consensus_response=self._merge_responses(largest) if largest else "",
            verifiers=frozenset(r.responder for r in largest),
            dissenting=frozenset(r.responder for r in responses if r not in largest),
            scores={"cluster_count": len(clusters)},
        )

    # ───────────────────────────────────────────────────────────────────────
    #  CLUSTERING
    # ───────────────────────────────────────────────────────────────────────

    def _cluster_responses(
        self,
        responses: list[Response],
    ) -> list[list[Response]]:
        """
        Cluster responses by similarity.

        Returns list of clusters (groups of similar responses).
        """
        if not responses:
            return []

        clusters: list[list[Response]] = []
        assigned: set[int] = set()

        for i, response in enumerate(responses):
            if i in assigned:
                continue

            # Start new cluster
            cluster = [response]
            assigned.add(i)

            # Find similar responses
            for j, other in enumerate(responses):
                if j in assigned:
                    continue

                similarity = self._text_similarity(response.text, other.text)
                if similarity >= self._similarity_threshold:
                    cluster.append(other)
                    assigned.add(j)

            clusters.append(cluster)

        return clusters

    def _text_similarity(self, a: str, b: str) -> float:
        """Calculate text similarity (0.0-1.0)."""
        if not a or not b:
            return 0.0

        # Use SequenceMatcher for robust similarity
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    # ───────────────────────────────────────────────────────────────────────
    #  MERGING
    # ───────────────────────────────────────────────────────────────────────

    def _merge_responses(self, responses: list[Response]) -> str:
        """
        Merge cluster of responses into consensus.

        Strategy: Use highest confidence response as base.
        (Future: semantic merging, extractive summarization)
        """
        if not responses:
            return ""

        # For now, use highest confidence
        best = max(responses, key=lambda r: r.confidence)
        return best.text

    def _aggregate_scores(
        self,
        responses: list[Response],
    ) -> dict[str, float]:
        """Aggregate verification scores from responses."""
        if not responses:
            return {}

        # Average confidence
        avg_confidence = sum(r.confidence for r in responses) / len(responses)

        return {
            "avg_confidence": avg_confidence,
            "verifier_count": len(responses),
        }

    # ───────────────────────────────────────────────────────────────────────
    #  VOTING
    # ───────────────────────────────────────────────────────────────────────

    async def vote(
        self,
        query: Query,
        response: Response,
    ) -> float:
        """
        Vote on a response's quality.

        Returns vote score (0.0-1.0).
        """
        # TODO: Implement DS-STAR scoring
        # For now, just return the response's confidence
        return response.confidence

    async def check_agreement(
        self,
        responses: list[Response],
        threshold: float = 0.66,
    ) -> bool:
        """Check if responses agree above threshold."""
        if len(responses) < 2:
            return len(responses) == 1

        clusters = self._cluster_responses(responses)
        if not clusters:
            return False

        largest = max(clusters, key=len)
        return len(largest) / len(responses) >= threshold


# ═══════════════════════════════════════════════════════════════════════════
#  DS-STAR SCORER - Quality assessment
# ═══════════════════════════════════════════════════════════════════════════


class DSStarScorer:
    """
    DS-STAR quality scoring for responses.

    D - Domain: Is the response relevant to the domain?
    S - Sensibility: Does the response make logical sense?
    T - Temporal: Is the information temporally accurate?
    A - Argument: Is the reasoning sound?
    R - Reference: Are sources credible?

    Usage:
        scorer = DSStarScorer()
        score = await scorer.score(query, response)
        print(f"Total: {score.total:.2f}")
    """

    def __init__(
        self,
        *,
        domain_keywords: dict[str, set[str]] | None = None,
    ):
        self._domain_keywords = domain_keywords or {}

    async def score(
        self,
        query: Query,
        response: Response,
    ) -> VerificationScore:
        """
        Score a response using DS-STAR.

        All scores are 0.0-1.0.
        """
        return VerificationScore(
            domain=await self._score_domain(query, response),
            sensibility=await self._score_sensibility(query, response),
            temporal=await self._score_temporal(query, response),
            argument=await self._score_argument(query, response),
            reference=await self._score_reference(query, response),
        )

    async def _score_domain(
        self,
        query: Query,
        response: Response,
    ) -> float:
        """Score domain relevance."""
        # Check if response mentions query terms
        query_terms = set(query.text.lower().split())
        response_terms = set(response.text.lower().split())

        overlap = len(query_terms & response_terms)
        if not query_terms:
            return 0.5

        return min(overlap / len(query_terms), 1.0)

    async def _score_sensibility(
        self,
        query: Query,
        response: Response,
    ) -> float:
        """Score logical sensibility."""
        # Basic heuristics
        text = response.text

        # Penalize very short or very long responses
        word_count = len(text.split())
        if word_count < 5:
            return 0.3
        if word_count > 1000:
            return 0.7

        # Check for contradiction markers
        contradiction_markers = ["but", "however", "although", "nevertheless"]
        contradictions = sum(1 for m in contradiction_markers if m in text.lower())

        # Some contradiction is okay (nuance), too much is bad
        if contradictions > 3:
            return 0.6

        return 0.8  # Default sensible

    async def _score_temporal(
        self,
        query: Query,
        response: Response,
    ) -> float:
        """Score temporal accuracy."""
        # Check for temporal markers
        temporal_markers = ["now", "currently", "today", "recently", "latest"]
        has_temporal = any(m in response.text.lower() for m in temporal_markers)

        # Can't verify temporal accuracy without external data
        # Just check if response acknowledges temporality when relevant
        query_temporal = any(m in query.text.lower() for m in temporal_markers)

        if query_temporal and not has_temporal:
            return 0.6  # Query asked about current info, response didn't address

        return 0.8  # Default okay

    async def _score_argument(
        self,
        query: Query,
        response: Response,
    ) -> float:
        """Score argumentative quality."""
        text = response.text.lower()

        # Reasoning markers
        reasoning_markers = [
            "because", "therefore", "thus", "hence",
            "since", "as a result", "consequently",
        ]

        has_reasoning = any(m in text for m in reasoning_markers)

        # Structure markers
        structure_markers = [
            "first", "second", "finally",
            "additionally", "moreover", "furthermore",
        ]

        has_structure = any(m in text for m in structure_markers)

        score = 0.5
        if has_reasoning:
            score += 0.25
        if has_structure:
            score += 0.25

        return score

    async def _score_reference(
        self,
        query: Query,
        response: Response,
    ) -> float:
        """Score reference quality."""
        text = response.text.lower()

        # Citation markers
        citation_markers = [
            "according to", "based on", "source",
            "research shows", "studies indicate",
        ]

        has_citation = any(m in text for m in citation_markers)

        if has_citation:
            return 0.8

        # No explicit references
        return 0.5
