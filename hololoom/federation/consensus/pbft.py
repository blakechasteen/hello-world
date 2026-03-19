"""
PBFT Consensus - Practical Byzantine Fault Tolerance.

Best for:
- Medium networks (100-1000 nodes)
- Byzantine fault tolerance required
- Strong consistency requirements

Characteristics:
- Immediate finality
- 3f+1 nodes required for f Byzantine faults
- O(n²) message complexity
- Leader-based with view changes

Reference:
    Castro & Liskov, "Practical Byzantine Fault Tolerance" (1999)
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

from .protocol import (
    ConsensusProtocol,
    ConsensusResult,
    FinalityType,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  PBFT PHASES
# ═══════════════════════════════════════════════════════════════════════════════


class PBFTPhase(Enum):
    """PBFT consensus phases."""

    IDLE = auto()
    PRE_PREPARE = auto()
    PREPARE = auto()
    COMMIT = auto()
    REPLY = auto()


# ═══════════════════════════════════════════════════════════════════════════════
#  PBFT MESSAGES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PBFTMessage:
    """Base PBFT message."""

    view: int
    sequence_number: int
    digest: str
    node_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PrePrepareMessage(PBFTMessage):
    """Pre-prepare message from primary."""

    request: Any = None


@dataclass
class PrepareMessage(PBFTMessage):
    """Prepare message from replicas."""
    pass


@dataclass
class CommitMessage(PBFTMessage):
    """Commit message from replicas."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  PBFT PROPOSAL
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PBFTProposal:
    """Internal tracking for a PBFT proposal."""

    proposal_id: str
    value: Any
    digest: str
    sequence_number: int
    view: int
    phase: PBFTPhase = PBFTPhase.IDLE
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Message collections
    pre_prepare: PrePrepareMessage | None = None
    prepares: dict[str, PrepareMessage] = field(default_factory=dict)
    commits: dict[str, CommitMessage] = field(default_factory=dict)

    # State
    prepared: bool = False
    committed: bool = False
    executed: bool = False
    result: ConsensusResult | None = None


# ═══════════════════════════════════════════════════════════════════════════════
#  PBFT CONSENSUS
# ═══════════════════════════════════════════════════════════════════════════════


class PBFTConsensus(ConsensusProtocol):
    """
    Practical Byzantine Fault Tolerance consensus.

    PBFT provides Byzantine fault tolerance with 3f+1 nodes for f faults.
    It uses a three-phase protocol (pre-prepare, prepare, commit) to ensure
    all non-faulty replicas agree on the total order of requests.

    Algorithm:
    1. PRE-PREPARE: Primary broadcasts <PRE-PREPARE, v, n, d> + request
    2. PREPARE: Replicas broadcast <PREPARE, v, n, d> when valid
    3. COMMIT: Replicas broadcast <COMMIT, v, n, d> when 2f+1 prepares seen
    4. REPLY: Replicas execute and reply when 2f+1 commits seen

    Properties:
    - Safety: All non-faulty replicas agree on request order
    - Liveness: Guaranteed progress (with view changes)
    - Immediate finality: Once committed, decision is final

    Configuration:
        node_id: This node's ID
        total_nodes: Total nodes in network (n = 3f+1)
        is_primary: Whether this node is the primary (leader)
        view: Current view number

    Usage:
        pbft = PBFTConsensus(
            node_id="node_1",
            total_nodes=7,  # 3f+1 where f=2
            is_primary=True,
        )

        proposal_id = await pbft.propose(my_value)

        # Collect votes (prepare and commit messages)
        for node in other_nodes:
            msg = await node.get_prepare(proposal_id)
            await pbft.receive_prepare(msg)

        result = await pbft.finalize(proposal_id)
    """

    def __init__(
        self,
        *,
        node_id: str = "primary",
        total_nodes: int = 4,
        is_primary: bool = True,
        view: int = 0,
        timeout_seconds: float = 30.0,
    ):
        self._node_id = node_id
        self._total_nodes = total_nodes
        self._is_primary = is_primary
        self._view = view
        self._timeout = timeout_seconds

        # Derived values
        self._f = (total_nodes - 1) // 3  # Max Byzantine faults
        self._quorum = 2 * self._f + 1    # 2f+1 for prepare/commit

        # State
        self._sequence_number = 0
        self._proposals: dict[str, PBFTProposal] = {}

        logger.debug(
            f"PBFT initialized: n={total_nodes}, f={self._f}, quorum={self._quorum}"
        )

    # ───────────────────────────────────────────────────────────────────────────
    #  PROTOCOL PROPERTIES
    # ───────────────────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "pbft"

    @property
    def max_byzantine_faults(self) -> int:
        return self._f

    @property
    def finality_type(self) -> FinalityType:
        return FinalityType.INSTANT

    @property
    def min_nodes(self) -> int:
        return 4  # 3f+1 where f=1

    @property
    def max_nodes(self) -> int:
        # O(n²) message complexity limits scalability
        return 1000

    # ───────────────────────────────────────────────────────────────────────────
    #  HELPER METHODS
    # ───────────────────────────────────────────────────────────────────────────

    def _compute_digest(self, value: Any) -> str:
        """Compute digest (hash) of a value."""
        content = str(value).encode("utf-8")
        return hashlib.sha256(content).hexdigest()[:16]

    # ───────────────────────────────────────────────────────────────────────────
    #  PROTOCOL METHODS
    # ───────────────────────────────────────────────────────────────────────────

    async def propose(self, value: Any) -> str:
        """
        Propose a value for consensus (PRE-PREPARE phase).

        Only the primary should call this. Backups should receive
        the pre-prepare message instead.
        """
        if not self._is_primary:
            raise ValueError("Only primary can propose")

        proposal_id = str(uuid.uuid4())
        self._sequence_number += 1
        digest = self._compute_digest(value)

        # Create proposal
        proposal = PBFTProposal(
            proposal_id=proposal_id,
            value=value,
            digest=digest,
            sequence_number=self._sequence_number,
            view=self._view,
            phase=PBFTPhase.PRE_PREPARE,
        )

        # Create pre-prepare message
        proposal.pre_prepare = PrePrepareMessage(
            view=self._view,
            sequence_number=self._sequence_number,
            digest=digest,
            node_id=self._node_id,
            request=value,
        )

        self._proposals[proposal_id] = proposal

        logger.debug(f"PBFT propose: {proposal_id[:8]}, seq={self._sequence_number}")
        return proposal_id

    async def receive_pre_prepare(
        self,
        proposal_id: str,
        msg: PrePrepareMessage,
    ) -> bool:
        """
        Receive a pre-prepare message (backup nodes).

        Returns True if message is valid and accepted.
        """
        # Validate message
        if msg.view != self._view:
            logger.warning(f"Invalid view in pre-prepare: {msg.view} != {self._view}")
            return False

        # Compute digest of request
        computed_digest = self._compute_digest(msg.request)
        if computed_digest != msg.digest:
            logger.warning("Digest mismatch in pre-prepare")
            return False

        # Create proposal tracking
        proposal = PBFTProposal(
            proposal_id=proposal_id,
            value=msg.request,
            digest=msg.digest,
            sequence_number=msg.sequence_number,
            view=msg.view,
            phase=PBFTPhase.PRE_PREPARE,
            pre_prepare=msg,
        )
        self._proposals[proposal_id] = proposal

        return True

    async def vote(self, proposal_id: str, vote: bool, node_id: str) -> None:
        """
        Cast a vote on a proposal.

        In PBFT, this translates to sending a PREPARE message.
        """
        if proposal_id not in self._proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")

        proposal = self._proposals[proposal_id]

        if vote:
            # Create prepare message
            prepare = PrepareMessage(
                view=proposal.view,
                sequence_number=proposal.sequence_number,
                digest=proposal.digest,
                node_id=node_id,
            )
            await self.receive_prepare(proposal_id, prepare)

    async def receive_prepare(
        self,
        proposal_id: str,
        msg: PrepareMessage,
    ) -> bool:
        """
        Receive a PREPARE message.

        Returns True if message is valid and accepted.
        """
        if proposal_id not in self._proposals:
            return False

        proposal = self._proposals[proposal_id]

        # Validate
        if msg.view != proposal.view:
            return False
        if msg.digest != proposal.digest:
            return False

        # Store prepare
        proposal.prepares[msg.node_id] = msg
        proposal.phase = PBFTPhase.PREPARE

        # Check if we have 2f+1 prepares (prepared predicate)
        if len(proposal.prepares) >= self._quorum and not proposal.prepared:
            proposal.prepared = True
            logger.debug(f"PBFT prepared: {proposal_id[:8]}")

            # Automatically send commit
            commit = CommitMessage(
                view=proposal.view,
                sequence_number=proposal.sequence_number,
                digest=proposal.digest,
                node_id=self._node_id,
            )
            await self.receive_commit(proposal_id, commit)

        return True

    async def receive_commit(
        self,
        proposal_id: str,
        msg: CommitMessage,
    ) -> bool:
        """
        Receive a COMMIT message.

        Returns True if message is valid and accepted.
        """
        if proposal_id not in self._proposals:
            return False

        proposal = self._proposals[proposal_id]

        # Validate
        if msg.view != proposal.view:
            return False
        if msg.digest != proposal.digest:
            return False

        # Store commit
        proposal.commits[msg.node_id] = msg
        proposal.phase = PBFTPhase.COMMIT

        # Check if we have 2f+1 commits (committed predicate)
        if len(proposal.commits) >= self._quorum and not proposal.committed:
            proposal.committed = True
            logger.debug(f"PBFT committed: {proposal_id[:8]}")

        return True

    async def finalize(self, proposal_id: str) -> ConsensusResult:
        """
        Finalize consensus on a proposal.

        This should be called after the commit phase completes.
        """
        if proposal_id not in self._proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")

        proposal = self._proposals[proposal_id]

        if proposal.result:
            return proposal.result

        # Check if committed
        n_prepares = len(proposal.prepares)
        n_commits = len(proposal.commits)

        if proposal.committed:
            result = ConsensusResult(
                proposal_id=proposal_id,
                finalized=True,
                value=proposal.value,
                round_number=proposal.sequence_number,
                votes_for=n_commits,
                votes_against=0,
                finality_type=self.finality_type,
                confidence=n_commits / self._total_nodes,
                agreeing_nodes=frozenset(proposal.commits.keys()),
                metadata={
                    "view": proposal.view,
                    "sequence_number": proposal.sequence_number,
                    "prepares": n_prepares,
                    "commits": n_commits,
                },
            )
        else:
            result = ConsensusResult(
                proposal_id=proposal_id,
                finalized=False,
                value=None,
                round_number=proposal.sequence_number,
                votes_for=n_commits,
                votes_against=0,
                finality_type=self.finality_type,
                confidence=n_commits / self._total_nodes if self._total_nodes > 0 else 0.0,
                metadata={
                    "view": proposal.view,
                    "phase": proposal.phase.name,
                    "prepares": n_prepares,
                    "commits": n_commits,
                    "quorum_needed": self._quorum,
                },
            )

        proposal.result = result
        proposal.executed = True
        proposal.phase = PBFTPhase.REPLY

        return result

    async def get_state(self) -> dict[str, Any]:
        """Get current protocol state."""
        base = await super().get_state()
        base.update({
            "node_id": self._node_id,
            "is_primary": self._is_primary,
            "view": self._view,
            "sequence_number": self._sequence_number,
            "f": self._f,
            "quorum": self._quorum,
            "active_proposals": len([p for p in self._proposals.values() if not p.executed]),
        })
        return base

    # ───────────────────────────────────────────────────────────────────────────
    #  VIEW CHANGE (simplified)
    # ───────────────────────────────────────────────────────────────────────────

    async def trigger_view_change(self) -> int:
        """
        Trigger a view change (leader rotation).

        This is a simplified implementation - full PBFT has more complex
        view change protocol for Byzantine safety.

        Returns:
            New view number
        """
        self._view += 1
        # In round-robin, primary is view % n
        new_primary_idx = self._view % self._total_nodes
        self._is_primary = (new_primary_idx == 0)  # Simplified

        logger.info(f"PBFT view change: view={self._view}, is_primary={self._is_primary}")
        return self._view
