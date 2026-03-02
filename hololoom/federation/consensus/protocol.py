"""
Consensus Protocol - Pluggable consensus algorithms.

Inspired by crypto consensus mechanisms:
- Quorum (simple, <100 nodes)
- Raft (crash-only failures, <100 nodes)
- PBFT (Byzantine faults, 100-1000 nodes)
- Avalanche (probabilistic, 1000+ nodes)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, Generic, List, Literal, Optional, TypeVar

from hololoom.federation.types import Response


# ═══════════════════════════════════════════════════════════════════════════════
#  FINALITY TYPES
# ═══════════════════════════════════════════════════════════════════════════════


class FinalityType(Enum):
    """How consensus decisions are finalized."""

    INSTANT = auto()        # Immediate finality (PBFT, Raft)
    PROBABILISTIC = auto()  # Probabilistic finality (Avalanche)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSENSUS RESULT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ConsensusResult:
    """Result of a consensus round."""

    proposal_id: str
    finalized: bool
    value: Any
    round_number: int = 0
    votes_for: int = 0
    votes_against: int = 0
    abstentions: int = 0
    finality_type: FinalityType = FinalityType.INSTANT
    confidence: float = 0.0  # For probabilistic finality
    agreeing_nodes: FrozenSet[str] = field(default_factory=frozenset)
    dissenting_nodes: FrozenSet[str] = field(default_factory=frozenset)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def agreement_ratio(self) -> float:
        """Ratio of agreeing votes to total votes."""
        total = self.votes_for + self.votes_against + self.abstentions
        if total == 0:
            return 0.0
        return self.votes_for / total


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSENSUS PROTOCOL ABC
# ═══════════════════════════════════════════════════════════════════════════════


class ConsensusProtocol(ABC):
    """
    Pluggable consensus protocol - swap algorithms based on network conditions.

    Implementations:
    - QuorumConsensus: Simple threshold voting (<100 nodes)
    - RaftConsensus: Leader-based log replication (crash-only)
    - PBFTConsensus: Byzantine fault tolerant (100-1000 nodes)
    - AvalancheConsensus: Probabilistic sampling (1000+ nodes)

    Usage:
        protocol = QuorumConsensus(threshold=0.66)

        proposal_id = await protocol.propose(value)
        await protocol.vote(proposal_id, vote=True)
        result = await protocol.finalize(proposal_id)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Algorithm name."""
        ...

    @property
    @abstractmethod
    def max_byzantine_faults(self) -> int:
        """Maximum Byzantine faults tolerated (f in 3f+1)."""
        ...

    @property
    @abstractmethod
    def finality_type(self) -> FinalityType:
        """Type of finality provided."""
        ...

    @property
    @abstractmethod
    def min_nodes(self) -> int:
        """Minimum nodes required for this algorithm."""
        ...

    @property
    @abstractmethod
    def max_nodes(self) -> int:
        """Maximum recommended nodes for this algorithm."""
        ...

    @abstractmethod
    async def propose(self, value: Any) -> str:
        """
        Propose a value for consensus.

        Args:
            value: The value to achieve consensus on

        Returns:
            proposal_id for tracking
        """
        ...

    @abstractmethod
    async def vote(self, proposal_id: str, vote: bool, node_id: str) -> None:
        """
        Cast a vote on a proposal.

        Args:
            proposal_id: The proposal to vote on
            vote: True = agree, False = disagree
            node_id: ID of the voting node
        """
        ...

    @abstractmethod
    async def finalize(self, proposal_id: str) -> ConsensusResult:
        """
        Finalize consensus on a proposal.

        Args:
            proposal_id: The proposal to finalize

        Returns:
            ConsensusResult with agreement details
        """
        ...

    async def get_state(self) -> Dict[str, Any]:
        """Get current protocol state (for debugging/monitoring)."""
        return {
            "name": self.name,
            "finality_type": self.finality_type.name,
            "max_byzantine_faults": self.max_byzantine_faults,
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  RESPONSE CONSENSUS PROTOCOL (Specialized for HoloLoom)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ResponseConsensusResult(ConsensusResult):
    """Consensus result for Response objects."""

    consensus_response: Optional[Response] = None
    all_responses: List[Response] = field(default_factory=list)


class ResponseConsensusProtocol(ConsensusProtocol):
    """
    Specialized consensus protocol for HoloLoom Response objects.

    Extends base protocol with:
    - Response similarity checking
    - Response merging
    - Confidence aggregation
    """

    @abstractmethod
    async def propose_response(self, response: Response) -> str:
        """Propose a response for consensus."""
        ...

    @abstractmethod
    async def collect_responses(
        self,
        proposal_id: str,
        responses: List[Response],
    ) -> ResponseConsensusResult:
        """
        Collect multiple responses and determine consensus.

        Args:
            proposal_id: The original proposal
            responses: Responses from multiple nodes

        Returns:
            ResponseConsensusResult with merged consensus response
        """
        ...
