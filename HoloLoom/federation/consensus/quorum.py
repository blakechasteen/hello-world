"""
Quorum Consensus - Simple threshold-based voting.

Best for:
- Small networks (<100 nodes)
- Low latency requirements
- Simple crash fault tolerance

Characteristics:
- Immediate finality
- 2f+1 nodes required for f crash faults
- No Byzantine fault tolerance
"""

from __future__ import annotations

import asyncio
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, FrozenSet, List, Optional, Set

from HoloLoom.federation.types import Response

from .protocol import (
    ConsensusProtocol,
    ConsensusResult,
    FinalityType,
    ResponseConsensusProtocol,
    ResponseConsensusResult,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  PROPOSAL TRACKING
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Proposal:
    """Internal tracking for a proposal."""

    proposal_id: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    votes_for: Set[str] = field(default_factory=set)
    votes_against: Set[str] = field(default_factory=set)
    finalized: bool = False
    result: Optional[ConsensusResult] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  QUORUM CONSENSUS
# ═══════════════════════════════════════════════════════════════════════════════


class QuorumConsensus(ConsensusProtocol):
    """
    Simple threshold-based quorum consensus.

    How it works:
    1. A proposal is made
    2. Nodes vote yes/no
    3. If votes_for / total_votes >= threshold, consensus is reached

    Configuration:
        threshold: Minimum agreement ratio (default: 0.66 = 2/3)
        min_votes: Minimum votes required (default: 3)
        timeout_seconds: Voting timeout (default: 30)

    Usage:
        consensus = QuorumConsensus(threshold=0.66, min_votes=3)

        proposal_id = await consensus.propose(my_value)

        # Collect votes from nodes
        for node_id in nodes:
            vote = await node.evaluate(my_value)
            await consensus.vote(proposal_id, vote, node_id)

        result = await consensus.finalize(proposal_id)
        if result.finalized:
            print(f"Consensus: {result.value}")
    """

    def __init__(
        self,
        *,
        threshold: float = 0.66,
        min_votes: int = 3,
        timeout_seconds: float = 30.0,
    ):
        self._threshold = threshold
        self._min_votes = min_votes
        self._timeout = timeout_seconds
        self._proposals: Dict[str, Proposal] = {}

    # ───────────────────────────────────────────────────────────────────────────
    #  PROTOCOL PROPERTIES
    # ───────────────────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "quorum"

    @property
    def max_byzantine_faults(self) -> int:
        # Quorum only tolerates crash faults, not Byzantine
        return 0

    @property
    def finality_type(self) -> FinalityType:
        return FinalityType.INSTANT

    @property
    def min_nodes(self) -> int:
        return self._min_votes

    @property
    def max_nodes(self) -> int:
        # Works well up to ~100 nodes
        return 100

    # ───────────────────────────────────────────────────────────────────────────
    #  PROTOCOL METHODS
    # ───────────────────────────────────────────────────────────────────────────

    async def propose(self, value: Any) -> str:
        """Propose a value for consensus."""
        proposal_id = str(uuid.uuid4())
        self._proposals[proposal_id] = Proposal(
            proposal_id=proposal_id,
            value=value,
        )
        return proposal_id

    async def vote(self, proposal_id: str, vote: bool, node_id: str) -> None:
        """Cast a vote on a proposal."""
        if proposal_id not in self._proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")

        proposal = self._proposals[proposal_id]
        if proposal.finalized:
            raise ValueError(f"Proposal already finalized: {proposal_id}")

        if vote:
            proposal.votes_for.add(node_id)
            proposal.votes_against.discard(node_id)
        else:
            proposal.votes_against.add(node_id)
            proposal.votes_for.discard(node_id)

    async def finalize(self, proposal_id: str) -> ConsensusResult:
        """Finalize consensus on a proposal."""
        if proposal_id not in self._proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")

        proposal = self._proposals[proposal_id]

        if proposal.finalized and proposal.result:
            return proposal.result

        votes_for = len(proposal.votes_for)
        votes_against = len(proposal.votes_against)
        total_votes = votes_for + votes_against

        # Check quorum requirements
        if total_votes < self._min_votes:
            result = ConsensusResult(
                proposal_id=proposal_id,
                finalized=False,
                value=proposal.value,
                votes_for=votes_for,
                votes_against=votes_against,
                finality_type=self.finality_type,
                confidence=0.0,
                agreeing_nodes=frozenset(proposal.votes_for),
                dissenting_nodes=frozenset(proposal.votes_against),
                metadata={"error": f"insufficient_votes:{total_votes}/{self._min_votes}"},
            )
        else:
            agreement_ratio = votes_for / total_votes
            finalized = agreement_ratio >= self._threshold

            result = ConsensusResult(
                proposal_id=proposal_id,
                finalized=finalized,
                value=proposal.value if finalized else None,
                votes_for=votes_for,
                votes_against=votes_against,
                finality_type=self.finality_type,
                confidence=agreement_ratio,
                agreeing_nodes=frozenset(proposal.votes_for),
                dissenting_nodes=frozenset(proposal.votes_against),
            )

        proposal.finalized = True
        proposal.result = result
        return result

    async def get_state(self) -> Dict[str, Any]:
        """Get current protocol state."""
        base = await super().get_state()
        base.update({
            "threshold": self._threshold,
            "min_votes": self._min_votes,
            "timeout_seconds": self._timeout,
            "active_proposals": len([p for p in self._proposals.values() if not p.finalized]),
        })
        return base


# ═══════════════════════════════════════════════════════════════════════════════
#  RESPONSE QUORUM CONSENSUS
# ═══════════════════════════════════════════════════════════════════════════════


class ResponseQuorumConsensus(ResponseConsensusProtocol):
    """
    Quorum consensus specialized for HoloLoom Response objects.

    Extends QuorumConsensus with:
    - Response similarity clustering
    - Response merging (pick highest confidence from agreeing cluster)
    - Confidence aggregation

    This is the default consensus algorithm for HoloLoom federation.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.66,
        min_votes: int = 3,
        similarity_threshold: float = 0.8,
        min_confidence: float = 0.3,
    ):
        self._threshold = threshold
        self._min_votes = min_votes
        self._similarity_threshold = similarity_threshold
        self._min_confidence = min_confidence
        self._proposals: Dict[str, Proposal] = {}
        self._responses: Dict[str, List[Response]] = {}

    # ───────────────────────────────────────────────────────────────────────────
    #  PROTOCOL PROPERTIES
    # ───────────────────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "response_quorum"

    @property
    def max_byzantine_faults(self) -> int:
        return 0

    @property
    def finality_type(self) -> FinalityType:
        return FinalityType.INSTANT

    @property
    def min_nodes(self) -> int:
        return self._min_votes

    @property
    def max_nodes(self) -> int:
        return 100

    # ───────────────────────────────────────────────────────────────────────────
    #  PROTOCOL METHODS
    # ───────────────────────────────────────────────────────────────────────────

    async def propose(self, value: Any) -> str:
        """Propose a value for consensus."""
        proposal_id = str(uuid.uuid4())
        self._proposals[proposal_id] = Proposal(
            proposal_id=proposal_id,
            value=value,
        )
        self._responses[proposal_id] = []
        return proposal_id

    async def propose_response(self, response: Response) -> str:
        """Propose a response for consensus."""
        return await self.propose(response)

    async def vote(self, proposal_id: str, vote: bool, node_id: str) -> None:
        """Cast a vote on a proposal."""
        if proposal_id not in self._proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")

        proposal = self._proposals[proposal_id]
        if vote:
            proposal.votes_for.add(node_id)
        else:
            proposal.votes_against.add(node_id)

    async def finalize(self, proposal_id: str) -> ConsensusResult:
        """Finalize consensus on a proposal."""
        # Use collect_responses for full logic
        responses = self._responses.get(proposal_id, [])
        return await self.collect_responses(proposal_id, responses)

    async def collect_responses(
        self,
        proposal_id: str,
        responses: List[Response],
    ) -> ResponseConsensusResult:
        """
        Collect multiple responses and determine consensus.

        This is the main entry point for response consensus.
        """
        # Filter low-confidence responses
        valid = [r for r in responses if r.confidence >= self._min_confidence]

        if len(valid) < self._min_votes:
            return ResponseConsensusResult(
                proposal_id=proposal_id,
                finalized=False,
                value=None,
                votes_for=len(valid),
                finality_type=self.finality_type,
                confidence=0.0,
                all_responses=responses,
                metadata={"error": f"insufficient_responses:{len(valid)}/{self._min_votes}"},
            )

        # Cluster similar responses
        clusters = self._cluster_responses(valid)

        if not clusters:
            return ResponseConsensusResult(
                proposal_id=proposal_id,
                finalized=False,
                value=None,
                finality_type=self.finality_type,
                confidence=0.0,
                all_responses=responses,
            )

        # Find largest agreeing cluster
        largest = max(clusters, key=len)
        agreement_ratio = len(largest) / len(valid)

        if agreement_ratio >= self._threshold:
            # Consensus reached
            consensus_response = self._merge_responses(largest)
            verifiers = frozenset(r.responder for r in largest)
            dissenting = frozenset(r.responder for r in valid if r.responder not in verifiers)

            return ResponseConsensusResult(
                proposal_id=proposal_id,
                finalized=True,
                value=consensus_response.text if consensus_response else None,
                votes_for=len(largest),
                votes_against=len(valid) - len(largest),
                finality_type=self.finality_type,
                confidence=agreement_ratio,
                agreeing_nodes=verifiers,
                dissenting_nodes=dissenting,
                consensus_response=consensus_response,
                all_responses=responses,
            )
        else:
            # No consensus
            return ResponseConsensusResult(
                proposal_id=proposal_id,
                finalized=False,
                value=None,
                votes_for=len(largest),
                votes_against=len(valid) - len(largest),
                finality_type=self.finality_type,
                confidence=agreement_ratio,
                all_responses=responses,
                metadata={"cluster_count": len(clusters)},
            )

    # ───────────────────────────────────────────────────────────────────────────
    #  CLUSTERING & MERGING
    # ───────────────────────────────────────────────────────────────────────────

    def _cluster_responses(self, responses: List[Response]) -> List[List[Response]]:
        """Cluster responses by text similarity."""
        if not responses:
            return []

        clusters: List[List[Response]] = []
        assigned: Set[int] = set()

        for i, response in enumerate(responses):
            if i in assigned:
                continue

            cluster = [response]
            assigned.add(i)

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
        """Calculate text similarity using SequenceMatcher."""
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _merge_responses(self, responses: List[Response]) -> Optional[Response]:
        """Merge cluster of responses - pick highest confidence."""
        if not responses:
            return None
        return max(responses, key=lambda r: r.confidence)
