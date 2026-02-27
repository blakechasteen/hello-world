"""
Federation Alignment Consensus Engine - Byzantine agreement on alignment.

Multi-node consensus for high-risk or disputed alignment decisions.
Implements Byzantine fault tolerance for distributed AI safety.

Philosophy: When in doubt, get multiple opinions. Trust emerges from agreement.

Key Features:
- Quorum-based voting (configurable by guild trust level)
- Byzantine fault detection (disagreement analysis)
- Weighted voting based on node trust scores
- Automatic escalation to human review

Quorum Requirements by Trust Level:
- STARTER guild: 5 verifiers (high paranoia)
- ESTABLISHED guild: 3 verifiers
- VETERAN guild: 2 verifiers
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

from .protocol import (
    AlignmentLayer,
    AlignmentStatus,
    AlignmentVerification,
    ConsensusAlignmentResult,
)
from .metadata import AlignmentMetadata, AlignmentProof
from ..types import Response, GuildTrustLevel


# ═══════════════════════════════════════════════════════════════════════════
#  CONSENSUS STRATEGY - How to reach agreement
# ═══════════════════════════════════════════════════════════════════════════


class ConsensusStrategy(Enum):
    """Strategy for reaching alignment consensus."""

    SIMPLE_MAJORITY = auto()      # >50% must agree
    SUPERMAJORITY = auto()        # >66% must agree (2/3)
    UNANIMOUS = auto()            # 100% must agree
    WEIGHTED = auto()             # Weight by trust score
    BYZANTINE_TOLERANT = auto()   # Can tolerate f < n/3 malicious


class ConsensusOutcome(Enum):
    """Outcome of consensus process."""

    CONSENSUS_VERIFIED = "consensus_verified"      # Agreement: aligned
    CONSENSUS_FAILED = "consensus_failed"          # Agreement: not aligned
    NO_CONSENSUS = "no_consensus"                  # No agreement reached
    ESCALATED = "escalated"                        # Requires human review
    TIMEOUT = "timeout"                            # Consensus timed out


# ═══════════════════════════════════════════════════════════════════════════
#  VOTE - Individual verifier vote
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class AlignmentVote:
    """A single verifier's vote on alignment."""

    voter_id: str                     # Node that voted
    vote: AlignmentStatus             # Their verdict
    confidence: float                 # Confidence in vote
    trust_weight: float               # Voter's trust score
    reasoning: List[str]              # Why they voted this way
    verification: AlignmentVerification  # Full verification result
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_positive(self) -> bool:
        """Is this a positive (verified) vote?"""
        return self.vote == AlignmentStatus.VERIFIED

    @property
    def weighted_vote(self) -> float:
        """Vote weighted by trust and confidence."""
        sign = 1.0 if self.is_positive else -1.0
        return sign * self.trust_weight * self.confidence


# ═══════════════════════════════════════════════════════════════════════════
#  CONSENSUS ENGINE - Multi-node agreement
# ═══════════════════════════════════════════════════════════════════════════


class AlignmentConsensusEngine:
    """
    Multi-node consensus engine for alignment verification.

    Coordinates multiple verifier nodes to reach Byzantine agreement
    on whether a response is aligned.

    Usage:
        engine = AlignmentConsensusEngine(
            strategy=ConsensusStrategy.BYZANTINE_TOLERANT,
            timeout_ms=500
        )

        result = await engine.reach_consensus(
            response=response,
            verifier_nodes=["node_1", "node_2", "node_3"],
            verify_func=verifier.verify_node_alignment,
            node_trust_scores={"node_1": 0.9, "node_2": 0.8, "node_3": 0.7}
        )

        if result.outcome == ConsensusOutcome.CONSENSUS_VERIFIED:
            # Safe to accept response
            ...
    """

    def __init__(
        self,
        strategy: ConsensusStrategy = ConsensusStrategy.BYZANTINE_TOLERANT,
        timeout_ms: float = 500.0,
        min_quorum: int = 2,
        max_byzantine_faults: Optional[int] = None,
    ):
        """
        Initialize consensus engine.

        Args:
            strategy: How to determine consensus
            timeout_ms: Maximum time to wait for consensus
            min_quorum: Minimum verifiers needed
            max_byzantine_faults: Max tolerable malicious nodes (default: n/3)
        """
        self.strategy = strategy
        self.timeout_ms = timeout_ms
        self.min_quorum = min_quorum
        self.max_byzantine_faults = max_byzantine_faults

    async def reach_consensus(
        self,
        response: Response,
        verifier_nodes: List[str],
        verify_func: Callable[
            [str, Response, float, AlignmentLayer],
            Coroutine[Any, Any, AlignmentVerification]
        ],
        node_trust_scores: Optional[Dict[str, float]] = None,
        alignment_threshold: float = 0.8,
    ) -> ConsensusResult:
        """
        Coordinate verifiers to reach alignment consensus.

        Args:
            response: Response to verify
            verifier_nodes: List of verifier node IDs
            verify_func: Function to call for each verification
            node_trust_scores: Trust scores per node (default: 0.5)
            alignment_threshold: Confidence threshold for verification

        Returns:
            ConsensusResult with outcome and details
        """
        start_time = time.perf_counter()
        node_trust_scores = node_trust_scores or {}

        # Validate quorum
        if len(verifier_nodes) < self.min_quorum:
            return ConsensusResult(
                request_id=response.request_id,
                outcome=ConsensusOutcome.NO_CONSENSUS,
                votes=[],
                agreement_ratio=0.0,
                consensus_confidence=0.0,
                quorum_met=False,
                quorum_required=self.min_quorum,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                reasoning=["Insufficient verifier nodes for quorum"],
            )

        # Collect votes from all verifiers (with timeout)
        votes = await self._collect_votes(
            response=response,
            verifier_nodes=verifier_nodes,
            verify_func=verify_func,
            node_trust_scores=node_trust_scores,
            alignment_threshold=alignment_threshold,
        )

        # Calculate consensus based on strategy
        outcome, agreement_ratio, confidence = self._calculate_consensus(votes)

        # Check for Byzantine behavior
        byzantine_detected = self._detect_byzantine_behavior(votes)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return ConsensusResult(
            request_id=response.request_id,
            outcome=outcome,
            votes=votes,
            agreement_ratio=agreement_ratio,
            consensus_confidence=confidence,
            quorum_met=len(votes) >= self.min_quorum,
            quorum_required=self.min_quorum,
            byzantine_detected=byzantine_detected,
            latency_ms=latency_ms,
            reasoning=self._generate_reasoning(votes, outcome),
        )

    async def _collect_votes(
        self,
        response: Response,
        verifier_nodes: List[str],
        verify_func: Callable,
        node_trust_scores: Dict[str, float],
        alignment_threshold: float,
    ) -> List[AlignmentVote]:
        """Collect votes from all verifier nodes."""
        votes: List[AlignmentVote] = []

        # Create verification tasks
        async def get_vote(node_id: str) -> Optional[AlignmentVote]:
            try:
                verification = await asyncio.wait_for(
                    verify_func(
                        node_id,
                        response,
                        alignment_threshold,
                        AlignmentLayer.L3_CONSENSUS,
                    ),
                    timeout=self.timeout_ms / 1000.0,
                )

                return AlignmentVote(
                    voter_id=node_id,
                    vote=verification.status,
                    confidence=verification.confidence,
                    trust_weight=node_trust_scores.get(node_id, 0.5),
                    reasoning=verification.reasoning_chain,
                    verification=verification,
                )
            except asyncio.TimeoutError:
                return None
            except Exception:
                return None

        # Collect votes concurrently
        tasks = [get_vote(node_id) for node_id in verifier_nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, AlignmentVote):
                votes.append(result)

        return votes

    def _calculate_consensus(
        self,
        votes: List[AlignmentVote],
    ) -> Tuple[ConsensusOutcome, float, float]:
        """
        Calculate consensus outcome based on strategy.

        Returns:
            (outcome, agreement_ratio, confidence)
        """
        if not votes:
            return ConsensusOutcome.TIMEOUT, 0.0, 0.0

        positive_votes = [v for v in votes if v.is_positive]
        negative_votes = [v for v in votes if not v.is_positive]

        n_positive = len(positive_votes)
        n_total = len(votes)
        agreement_ratio = n_positive / n_total

        if self.strategy == ConsensusStrategy.SIMPLE_MAJORITY:
            threshold = 0.5
        elif self.strategy == ConsensusStrategy.SUPERMAJORITY:
            threshold = 2.0 / 3.0
        elif self.strategy == ConsensusStrategy.UNANIMOUS:
            threshold = 1.0
        elif self.strategy == ConsensusStrategy.WEIGHTED:
            return self._calculate_weighted_consensus(votes)
        elif self.strategy == ConsensusStrategy.BYZANTINE_TOLERANT:
            return self._calculate_byzantine_consensus(votes)
        else:
            threshold = 0.5

        # Calculate confidence as average of agreeing votes
        if agreement_ratio >= threshold:
            confidence = (
                sum(v.confidence for v in positive_votes) / n_positive
                if positive_votes
                else 0.0
            )
            return ConsensusOutcome.CONSENSUS_VERIFIED, agreement_ratio, confidence
        elif (1 - agreement_ratio) >= threshold:
            confidence = (
                sum(v.confidence for v in negative_votes) / len(negative_votes)
                if negative_votes
                else 0.0
            )
            return ConsensusOutcome.CONSENSUS_FAILED, agreement_ratio, confidence
        else:
            return ConsensusOutcome.NO_CONSENSUS, agreement_ratio, 0.5

    def _calculate_weighted_consensus(
        self,
        votes: List[AlignmentVote],
    ) -> Tuple[ConsensusOutcome, float, float]:
        """Calculate consensus weighted by trust scores."""
        if not votes:
            return ConsensusOutcome.TIMEOUT, 0.0, 0.0

        total_weight = sum(v.trust_weight for v in votes)
        weighted_sum = sum(v.weighted_vote for v in votes)

        if total_weight == 0:
            return ConsensusOutcome.NO_CONSENSUS, 0.0, 0.0

        normalized_score = (weighted_sum / total_weight + 1) / 2  # Scale to 0-1

        if normalized_score >= 0.6:  # Weighted majority
            confidence = sum(
                v.confidence * v.trust_weight for v in votes if v.is_positive
            ) / sum(v.trust_weight for v in votes if v.is_positive)
            return ConsensusOutcome.CONSENSUS_VERIFIED, normalized_score, confidence
        elif normalized_score <= 0.4:
            confidence = sum(
                v.confidence * v.trust_weight for v in votes if not v.is_positive
            ) / sum(v.trust_weight for v in votes if not v.is_positive or 1)
            return ConsensusOutcome.CONSENSUS_FAILED, normalized_score, confidence
        else:
            return ConsensusOutcome.NO_CONSENSUS, normalized_score, 0.5

    def _calculate_byzantine_consensus(
        self,
        votes: List[AlignmentVote],
    ) -> Tuple[ConsensusOutcome, float, float]:
        """
        Byzantine fault tolerant consensus.

        Requires >2/3 agreement to tolerate f < n/3 malicious nodes.
        """
        if not votes:
            return ConsensusOutcome.TIMEOUT, 0.0, 0.0

        n = len(votes)
        max_faults = self.max_byzantine_faults or (n - 1) // 3
        required_agreement = n - max_faults  # Need n - f agreeing

        positive_votes = [v for v in votes if v.is_positive]
        n_positive = len(positive_votes)

        agreement_ratio = n_positive / n

        if n_positive >= required_agreement:
            confidence = sum(v.confidence for v in positive_votes) / n_positive
            return ConsensusOutcome.CONSENSUS_VERIFIED, agreement_ratio, confidence
        elif (n - n_positive) >= required_agreement:
            negative_votes = [v for v in votes if not v.is_positive]
            confidence = (
                sum(v.confidence for v in negative_votes) / len(negative_votes)
                if negative_votes
                else 0.0
            )
            return ConsensusOutcome.CONSENSUS_FAILED, agreement_ratio, confidence
        else:
            return ConsensusOutcome.NO_CONSENSUS, agreement_ratio, 0.5

    def _detect_byzantine_behavior(
        self,
        votes: List[AlignmentVote],
    ) -> List[str]:
        """
        Detect potential Byzantine (malicious) behavior.

        Returns list of suspicious node IDs.
        """
        suspicious: List[str] = []

        if len(votes) < 3:
            return suspicious

        # Check for outlier confidence levels
        confidences = [v.confidence for v in votes]
        avg_confidence = sum(confidences) / len(confidences)
        std_confidence = (
            sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        ) ** 0.5

        for vote in votes:
            # Flag if confidence is suspiciously different
            if abs(vote.confidence - avg_confidence) > 2 * std_confidence:
                suspicious.append(vote.voter_id)

            # Flag if low trust node strongly disagrees with high trust majority
            if vote.trust_weight < 0.3:
                high_trust_votes = [
                    v for v in votes if v.trust_weight >= 0.7 and v.voter_id != vote.voter_id
                ]
                if high_trust_votes:
                    high_trust_consensus = sum(
                        1 for v in high_trust_votes if v.is_positive
                    ) / len(high_trust_votes)
                    if (high_trust_consensus > 0.8 and not vote.is_positive) or (
                        high_trust_consensus < 0.2 and vote.is_positive
                    ):
                        suspicious.append(vote.voter_id)

        return list(set(suspicious))

    def _generate_reasoning(
        self,
        votes: List[AlignmentVote],
        outcome: ConsensusOutcome,
    ) -> List[str]:
        """Generate human-readable reasoning for consensus result."""
        reasoning: List[str] = []

        n_total = len(votes)
        n_positive = sum(1 for v in votes if v.is_positive)
        n_negative = n_total - n_positive

        reasoning.append(f"Received {n_total} votes: {n_positive} positive, {n_negative} negative")
        reasoning.append(f"Strategy: {self.strategy.name}")
        reasoning.append(f"Outcome: {outcome.value}")

        if votes:
            avg_confidence = sum(v.confidence for v in votes) / n_total
            reasoning.append(f"Average confidence: {avg_confidence:.2f}")

        return reasoning


# ═══════════════════════════════════════════════════════════════════════════
#  CONSENSUS RESULT - Full consensus outcome
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class ConsensusResult:
    """Complete result of consensus process."""

    request_id: str
    outcome: ConsensusOutcome
    votes: List[AlignmentVote]
    agreement_ratio: float
    consensus_confidence: float
    quorum_met: bool
    quorum_required: int
    latency_ms: float
    reasoning: List[str] = field(default_factory=list)
    byzantine_detected: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_verified(self) -> bool:
        """Was alignment consensus reached?"""
        return self.outcome == ConsensusOutcome.CONSENSUS_VERIFIED

    @property
    def requires_escalation(self) -> bool:
        """Does this need human review?"""
        return (
            self.outcome in (ConsensusOutcome.NO_CONSENSUS, ConsensusOutcome.ESCALATED)
            or len(self.byzantine_detected) > 0
            or not self.quorum_met
        )

    def to_consensus_alignment_result(self) -> ConsensusAlignmentResult:
        """Convert to protocol ConsensusAlignmentResult."""
        return ConsensusAlignmentResult(
            request_id=self.request_id,
            verified=self.is_verified,
            agreement_ratio=self.agreement_ratio,
            min_required=self.quorum_required,
            verifier_nodes=[v.voter_id for v in self.votes if v.is_positive],
            dissenting_nodes=[v.voter_id for v in self.votes if not v.is_positive],
            individual_results=[v.verification for v in self.votes],
            consensus_confidence=self.consensus_confidence,
            latency_ms=self.latency_ms,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def get_quorum_for_trust_level(trust_level: GuildTrustLevel) -> int:
    """Get required quorum based on guild trust level."""
    return {
        GuildTrustLevel.STARTER: 5,      # New guilds: high paranoia
        GuildTrustLevel.ESTABLISHED: 3,  # Proven guilds: moderate
        GuildTrustLevel.VETERAN: 2,      # Veteran guilds: minimal
    }.get(trust_level, 3)


def create_consensus_engine(
    trust_level: GuildTrustLevel = GuildTrustLevel.ESTABLISHED,
    strategy: ConsensusStrategy = ConsensusStrategy.BYZANTINE_TOLERANT,
    timeout_ms: float = 500.0,
) -> AlignmentConsensusEngine:
    """
    Create consensus engine with appropriate quorum for trust level.

    Args:
        trust_level: Guild trust level (determines quorum)
        strategy: Consensus strategy
        timeout_ms: Timeout for consensus

    Returns:
        Configured AlignmentConsensusEngine
    """
    quorum = get_quorum_for_trust_level(trust_level)

    return AlignmentConsensusEngine(
        strategy=strategy,
        timeout_ms=timeout_ms,
        min_quorum=quorum,
    )
