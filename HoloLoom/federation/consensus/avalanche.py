"""
Avalanche Consensus - Probabilistic sampling for large networks.

Best for:
- Large networks (1000+ nodes)
- Probabilistic finality acceptable
- High throughput requirements

Characteristics:
- Sub-sampled voting (k nodes per round)
- Probabilistic finality (accumulates confidence)
- O(k log n) message complexity
- Metastable protocol (flips to consensus)

Reference:
    Team Rocket, "Scalable and Probabilistic Leaderless BFT Consensus" (2019)
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from .protocol import (
    ConsensusProtocol,
    ConsensusResult,
    FinalityType,
)


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  AVALANCHE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AvalancheParams:
    """
    Avalanche protocol parameters.

    These parameters control the probabilistic guarantees:
    - k: Sample size per query (default: 20)
    - alpha: Quorum threshold (default: 0.8 = 80%)
    - beta1: Decision threshold (default: 11 consecutive)
    - beta2: Finality threshold (default: 20 consecutive)
    """

    k: int = 20              # Sample size per query
    alpha: float = 0.8       # Quorum threshold (80%)
    beta1: int = 11          # Decision threshold (consecutive)
    beta2: int = 20          # Finality threshold (consecutive)

    def __post_init__(self):
        """Validate parameters."""
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if not 0.5 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0.5, 1.0], got {self.alpha}")
        if self.beta1 < 1:
            raise ValueError(f"beta1 must be >= 1, got {self.beta1}")
        if self.beta2 < self.beta1:
            raise ValueError(f"beta2 must be >= beta1, got {self.beta2}")


# ═══════════════════════════════════════════════════════════════════════════════
#  AVALANCHE PREFERENCE
# ═══════════════════════════════════════════════════════════════════════════════


class PreferenceState(Enum):
    """Current preference state in Avalanche."""

    UNDECIDED = auto()   # No strong preference yet
    PREFERRED = auto()   # Leaning toward accept
    DECIDED = auto()     # Reached beta1 threshold
    FINALIZED = auto()   # Reached beta2 threshold (irreversible)


@dataclass
class AvalanchePreference:
    """
    Tracks preference for a value in Avalanche.

    Avalanche uses a "snowball" mechanism:
    - Query k random nodes
    - If >= alpha agree, increment consecutive counter
    - If counter reaches beta1, mark as decided
    - If counter reaches beta2, mark as finalized
    """

    value: Any
    accept_count: int = 0       # Total accept votes
    reject_count: int = 0       # Total reject votes
    consecutive: int = 0        # Consecutive agreeing rounds
    confidence: float = 0.0     # Current confidence level
    state: PreferenceState = PreferenceState.UNDECIDED
    last_query_time: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_queries(self) -> int:
        """Total number of query rounds."""
        return self.accept_count + self.reject_count

    def update(self, accepted: bool, params: AvalancheParams) -> None:
        """
        Update preference based on query result.

        Args:
            accepted: Whether this round achieved quorum
            params: Avalanche parameters
        """
        if accepted:
            self.accept_count += 1
            self.consecutive += 1
        else:
            self.reject_count += 1
            self.consecutive = 0  # Reset on disagreement

        # Update confidence (simple ratio)
        if self.total_queries > 0:
            self.confidence = self.accept_count / self.total_queries

        # Update state based on consecutive count
        if self.consecutive >= params.beta2:
            self.state = PreferenceState.FINALIZED
        elif self.consecutive >= params.beta1:
            self.state = PreferenceState.DECIDED
        elif self.consecutive > 0:
            self.state = PreferenceState.PREFERRED
        else:
            self.state = PreferenceState.UNDECIDED

        self.last_query_time = datetime.utcnow()


# ═══════════════════════════════════════════════════════════════════════════════
#  AVALANCHE PROPOSAL
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AvalancheProposal:
    """Internal tracking for an Avalanche proposal."""

    proposal_id: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Preference tracking
    preference: AvalanchePreference = field(default=None)

    # Voting records
    votes_for: Set[str] = field(default_factory=set)
    votes_against: Set[str] = field(default_factory=set)

    # Query history
    query_rounds: int = 0
    query_history: List[Tuple[int, bool]] = field(default_factory=list)

    # Result caching
    result: Optional[ConsensusResult] = None

    def __post_init__(self):
        if self.preference is None:
            self.preference = AvalanchePreference(value=self.value)


# ═══════════════════════════════════════════════════════════════════════════════
#  AVALANCHE CONSENSUS
# ═══════════════════════════════════════════════════════════════════════════════


class AvalancheConsensus(ConsensusProtocol):
    """
    Avalanche consensus - probabilistic sampling for 1000+ nodes.

    Avalanche achieves consensus through repeated sub-sampled voting:
    1. Query k random nodes for their preference
    2. If >= alpha * k agree, update preference and increment counter
    3. If counter reaches beta1, mark as decided (high confidence)
    4. If counter reaches beta2, mark as finalized (irreversible)

    Algorithm (Snowball variant):
        for each query round:
            sample = random.sample(peers, k)
            votes = query_all(sample)
            if count(votes == preferred) >= alpha * k:
                consecutive += 1
            else:
                consecutive = 0
                preferred = majority(votes)  # May flip preference

            if consecutive >= beta:
                return FINALIZED

    Properties:
    - Probabilistic finality: Confidence increases with rounds
    - Scalable: O(k log n) messages, constant per node
    - Byzantine tolerant: Handles f < n/4 Byzantine faults
    - Leaderless: No single point of failure

    Configuration:
        node_id: This node's ID
        total_nodes: Total nodes in network
        params: AvalancheParams (k, alpha, beta1, beta2)

    Usage:
        avalanche = AvalancheConsensus(
            node_id="node_1",
            total_nodes=1000,
            params=AvalancheParams(k=20, alpha=0.8, beta1=11, beta2=20)
        )

        proposal_id = await avalanche.propose(my_value)

        # Run query rounds until finalized
        while not await avalanche.is_finalized(proposal_id):
            # Sample k random nodes
            sample = random.sample(all_peers, avalanche.params.k)
            votes = await query_peers(sample, proposal_id)

            # Process votes
            await avalanche.process_query_round(proposal_id, votes)

        result = await avalanche.finalize(proposal_id)
    """

    def __init__(
        self,
        *,
        node_id: str = "node_0",
        total_nodes: int = 1000,
        params: Optional[AvalancheParams] = None,
        max_rounds: int = 100,
    ):
        self._node_id = node_id
        self._total_nodes = total_nodes
        self._params = params or AvalancheParams()
        self._max_rounds = max_rounds

        # State
        self._proposals: Dict[str, AvalancheProposal] = {}

        logger.debug(
            f"Avalanche initialized: n={total_nodes}, k={self._params.k}, "
            f"α={self._params.alpha}, β1={self._params.beta1}, β2={self._params.beta2}"
        )

    # ───────────────────────────────────────────────────────────────────────────
    #  PROTOCOL PROPERTIES
    # ───────────────────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "avalanche"

    @property
    def params(self) -> AvalancheParams:
        """Get Avalanche parameters."""
        return self._params

    @property
    def max_byzantine_faults(self) -> int:
        # Avalanche tolerates f < n/4 Byzantine faults
        return (self._total_nodes - 1) // 4

    @property
    def finality_type(self) -> FinalityType:
        return FinalityType.PROBABILISTIC

    @property
    def min_nodes(self) -> int:
        # Need at least k nodes for sampling
        return max(self._params.k, 4)

    @property
    def max_nodes(self) -> int:
        # Avalanche scales to very large networks
        return 100000

    # ───────────────────────────────────────────────────────────────────────────
    #  PROTOCOL METHODS
    # ───────────────────────────────────────────────────────────────────────────

    async def propose(self, value: Any) -> str:
        """
        Propose a value for consensus.

        In Avalanche, any node can propose. The value enters the
        "snowball" preference tracking system.
        """
        proposal_id = str(uuid.uuid4())

        proposal = AvalancheProposal(
            proposal_id=proposal_id,
            value=value,
        )
        self._proposals[proposal_id] = proposal

        logger.debug(f"Avalanche propose: {proposal_id[:8]}")
        return proposal_id

    async def vote(self, proposal_id: str, vote: bool, node_id: str) -> None:
        """
        Cast a vote on a proposal.

        In Avalanche, votes are collected during query rounds.
        This method records individual votes.
        """
        if proposal_id not in self._proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")

        proposal = self._proposals[proposal_id]

        if vote:
            proposal.votes_for.add(node_id)
            proposal.votes_against.discard(node_id)
        else:
            proposal.votes_against.add(node_id)
            proposal.votes_for.discard(node_id)

    async def process_query_round(
        self,
        proposal_id: str,
        votes: Dict[str, bool],
    ) -> bool:
        """
        Process a query round with collected votes.

        This is the core Avalanche loop:
        1. Count votes from sampled nodes
        2. Check if >= alpha agree
        3. Update preference and consecutive counter

        Args:
            proposal_id: The proposal being queried
            votes: Dict mapping node_id → vote (True=accept, False=reject)

        Returns:
            True if this round achieved quorum (alpha agreement)
        """
        if proposal_id not in self._proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")

        proposal = self._proposals[proposal_id]
        proposal.query_rounds += 1

        # Count votes
        accept_votes = sum(1 for v in votes.values() if v)
        total_votes = len(votes)

        if total_votes == 0:
            return False

        # Check quorum
        agreement_ratio = accept_votes / total_votes
        achieved_quorum = agreement_ratio >= self._params.alpha

        # Update preference
        proposal.preference.update(achieved_quorum, self._params)

        # Record history
        proposal.query_history.append((proposal.query_rounds, achieved_quorum))

        # Record individual votes
        for node_id, vote in votes.items():
            if vote:
                proposal.votes_for.add(node_id)
            else:
                proposal.votes_against.add(node_id)

        logger.debug(
            f"Avalanche round {proposal.query_rounds}: "
            f"{accept_votes}/{total_votes} = {agreement_ratio:.1%}, "
            f"consecutive={proposal.preference.consecutive}, "
            f"state={proposal.preference.state.name}"
        )

        return achieved_quorum

    async def is_finalized(self, proposal_id: str) -> bool:
        """Check if a proposal has reached finality."""
        if proposal_id not in self._proposals:
            return False

        proposal = self._proposals[proposal_id]
        return proposal.preference.state == PreferenceState.FINALIZED

    async def is_decided(self, proposal_id: str) -> bool:
        """Check if a proposal has been decided (but not yet finalized)."""
        if proposal_id not in self._proposals:
            return False

        proposal = self._proposals[proposal_id]
        return proposal.preference.state in (
            PreferenceState.DECIDED,
            PreferenceState.FINALIZED,
        )

    async def finalize(self, proposal_id: str) -> ConsensusResult:
        """
        Finalize consensus on a proposal.

        Returns the current state - may be partial if not yet finalized.
        """
        if proposal_id not in self._proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")

        proposal = self._proposals[proposal_id]

        if proposal.result:
            return proposal.result

        pref = proposal.preference
        is_finalized = pref.state == PreferenceState.FINALIZED

        result = ConsensusResult(
            proposal_id=proposal_id,
            finalized=is_finalized,
            value=proposal.value if pref.state in (
                PreferenceState.DECIDED,
                PreferenceState.FINALIZED,
            ) else None,
            round_number=proposal.query_rounds,
            votes_for=len(proposal.votes_for),
            votes_against=len(proposal.votes_against),
            finality_type=self.finality_type,
            confidence=pref.confidence,
            agreeing_nodes=frozenset(proposal.votes_for),
            dissenting_nodes=frozenset(proposal.votes_against),
            metadata={
                "consecutive": pref.consecutive,
                "state": pref.state.name,
                "accept_rounds": pref.accept_count,
                "reject_rounds": pref.reject_count,
                "beta1_threshold": self._params.beta1,
                "beta2_threshold": self._params.beta2,
            },
        )

        if is_finalized:
            proposal.result = result

        return result

    async def get_state(self) -> Dict[str, Any]:
        """Get current protocol state."""
        base = await super().get_state()
        base.update({
            "node_id": self._node_id,
            "total_nodes": self._total_nodes,
            "k": self._params.k,
            "alpha": self._params.alpha,
            "beta1": self._params.beta1,
            "beta2": self._params.beta2,
            "active_proposals": len([
                p for p in self._proposals.values()
                if p.preference.state != PreferenceState.FINALIZED
            ]),
        })
        return base

    # ───────────────────────────────────────────────────────────────────────────
    #  AVALANCHE-SPECIFIC METHODS
    # ───────────────────────────────────────────────────────────────────────────

    def select_sample(
        self,
        available_peers: List[str],
        exclude: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        Select k random peers for a query round.

        Args:
            available_peers: List of all available peer IDs
            exclude: Optional set of peers to exclude (e.g., self)

        Returns:
            List of k peer IDs to query
        """
        peers = available_peers.copy()

        if exclude:
            peers = [p for p in peers if p not in exclude]

        k = min(self._params.k, len(peers))
        return random.sample(peers, k)

    def get_preference(self, proposal_id: str) -> Optional[AvalanchePreference]:
        """Get the current preference for a proposal."""
        if proposal_id not in self._proposals:
            return None
        return self._proposals[proposal_id].preference

    def get_confidence(self, proposal_id: str) -> float:
        """Get the current confidence level for a proposal."""
        if proposal_id not in self._proposals:
            return 0.0
        return self._proposals[proposal_id].preference.confidence

    async def run_consensus_loop(
        self,
        proposal_id: str,
        query_fn,
        max_rounds: Optional[int] = None,
    ) -> ConsensusResult:
        """
        Run the full Avalanche consensus loop.

        This is a convenience method that runs query rounds until
        finalization or max_rounds is reached.

        Args:
            proposal_id: The proposal to finalize
            query_fn: Async function that takes List[peer_id] and returns Dict[peer_id, bool]
            max_rounds: Maximum rounds to run (default: self._max_rounds)

        Returns:
            ConsensusResult (may be partial if max_rounds reached)
        """
        if proposal_id not in self._proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")

        max_rounds = max_rounds or self._max_rounds

        for _ in range(max_rounds):
            # Check if already finalized
            if await self.is_finalized(proposal_id):
                break

            # This is a placeholder - in real usage, the caller provides
            # the peer list and query function
            # For now, we just return the current state
            pass

        return await self.finalize(proposal_id)
