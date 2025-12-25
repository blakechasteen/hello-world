"""
Raft Consensus - Leader-based log replication.

Best for:
- Small to medium networks (<100 nodes)
- Crash fault tolerance (no Byzantine faults)
- Strong consistency with leader election

Characteristics:
- Leader-based architecture
- Log replication for durability
- Term-based leader election
- 2f+1 nodes required for f crash faults

Reference:
    Ongaro & Ousterhout, "In Search of an Understandable Consensus Algorithm" (2014)
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set

from .protocol import (
    ConsensusProtocol,
    ConsensusResult,
    FinalityType,
)


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  RAFT NODE STATE
# ═══════════════════════════════════════════════════════════════════════════════


class RaftState(Enum):
    """Raft node states."""

    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


# ═══════════════════════════════════════════════════════════════════════════════
#  RAFT LOG ENTRIES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LogEntry:
    """Entry in the Raft log."""

    index: int
    term: int
    command: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AppendEntriesRequest:
    """AppendEntries RPC request (heartbeat + log replication)."""

    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: List[LogEntry]
    leader_commit: int


@dataclass
class AppendEntriesResponse:
    """AppendEntries RPC response."""

    term: int
    success: bool
    match_index: int = 0


@dataclass
class RequestVoteRequest:
    """RequestVote RPC request."""

    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int


@dataclass
class RequestVoteResponse:
    """RequestVote RPC response."""

    term: int
    vote_granted: bool


# ═══════════════════════════════════════════════════════════════════════════════
#  RAFT PROPOSAL
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RaftProposal:
    """Internal tracking for a Raft proposal."""

    proposal_id: str
    value: Any
    log_index: int
    term: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # State
    committed: bool = False
    applied: bool = False
    result: Optional[ConsensusResult] = None

    # Replication tracking
    ack_nodes: Set[str] = field(default_factory=set)


# ═══════════════════════════════════════════════════════════════════════════════
#  RAFT CONSENSUS
# ═══════════════════════════════════════════════════════════════════════════════


class RaftConsensus(ConsensusProtocol):
    """
    Raft consensus algorithm.

    Raft provides crash fault tolerance with 2f+1 nodes for f faults.
    It uses leader election and log replication for consensus.

    Algorithm:
    1. Leader Election: Candidates request votes, majority wins
    2. Log Replication: Leader appends entries, replicates to followers
    3. Commitment: Entry committed when replicated to majority
    4. Safety: Only leaders with up-to-date logs can be elected

    Properties:
    - Safety: Only one leader per term
    - Liveness: Guaranteed progress (with election timeouts)
    - Immediate finality: Once committed, decision is final
    - No Byzantine tolerance: Assumes nodes are honest but may crash

    Configuration:
        node_id: This node's ID
        total_nodes: Total nodes in cluster
        is_leader: Whether this node starts as leader
        election_timeout_range: (min, max) timeout in seconds

    Usage:
        raft = RaftConsensus(
            node_id="node_1",
            total_nodes=5,  # 2f+1 where f=2
        )

        proposal_id = await raft.propose(my_value)

        # Leader replicates to followers
        for node in followers:
            response = await node.append_entries(entries)
            await raft.receive_append_response(node.id, response)

        result = await raft.finalize(proposal_id)
    """

    def __init__(
        self,
        *,
        node_id: str = "node_0",
        total_nodes: int = 3,
        is_leader: bool = False,
        election_timeout_range: tuple = (0.15, 0.3),
        heartbeat_interval: float = 0.05,
    ):
        self._node_id = node_id
        self._total_nodes = total_nodes
        self._heartbeat_interval = heartbeat_interval
        self._election_timeout_range = election_timeout_range

        # Derived values
        self._majority = (total_nodes // 2) + 1

        # Persistent state (on stable storage in real implementation)
        self._current_term = 0
        self._voted_for: Optional[str] = None
        self._log: List[LogEntry] = []

        # Volatile state (all servers)
        self._commit_index = 0
        self._last_applied = 0
        self._state = RaftState.LEADER if is_leader else RaftState.FOLLOWER
        self._leader_id: Optional[str] = node_id if is_leader else None

        # Volatile state (leaders only)
        self._next_index: Dict[str, int] = {}
        self._match_index: Dict[str, int] = {}

        # Proposal tracking
        self._proposals: Dict[str, RaftProposal] = {}

        if is_leader:
            self._current_term = 1

        logger.debug(
            f"Raft initialized: n={total_nodes}, majority={self._majority}, "
            f"state={self._state.name}"
        )

    # ───────────────────────────────────────────────────────────────────────────
    #  PROTOCOL PROPERTIES
    # ───────────────────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "raft"

    @property
    def max_byzantine_faults(self) -> int:
        # Raft only tolerates crash faults, not Byzantine
        return 0

    @property
    def finality_type(self) -> FinalityType:
        return FinalityType.INSTANT

    @property
    def min_nodes(self) -> int:
        return 3  # 2f+1 where f=1

    @property
    def max_nodes(self) -> int:
        # Raft works well up to ~100 nodes
        return 100

    @property
    def is_leader(self) -> bool:
        return self._state == RaftState.LEADER

    @property
    def current_term(self) -> int:
        return self._current_term

    @property
    def commit_index(self) -> int:
        return self._commit_index

    # ───────────────────────────────────────────────────────────────────────────
    #  PROTOCOL METHODS
    # ───────────────────────────────────────────────────────────────────────────

    async def propose(self, value: Any) -> str:
        """
        Propose a value for consensus (leader only).

        The value is appended to the log and will be replicated
        to followers via AppendEntries RPCs.
        """
        if not self.is_leader:
            raise ValueError("Only leader can propose. Forward to leader.")

        proposal_id = str(uuid.uuid4())

        # Append to log
        log_index = len(self._log) + 1
        entry = LogEntry(
            index=log_index,
            term=self._current_term,
            command=value,
        )
        self._log.append(entry)

        # Create proposal tracking
        proposal = RaftProposal(
            proposal_id=proposal_id,
            value=value,
            log_index=log_index,
            term=self._current_term,
        )
        proposal.ack_nodes.add(self._node_id)  # Leader has it

        self._proposals[proposal_id] = proposal

        logger.debug(f"Raft propose: {proposal_id[:8]}, index={log_index}")
        return proposal_id

    async def vote(self, proposal_id: str, vote: bool, node_id: str) -> None:
        """
        Cast a vote on a proposal.

        In Raft, this translates to acknowledging log replication.
        """
        if proposal_id not in self._proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")

        proposal = self._proposals[proposal_id]

        if vote:
            proposal.ack_nodes.add(node_id)

            # Check if we have majority
            if len(proposal.ack_nodes) >= self._majority and not proposal.committed:
                proposal.committed = True
                self._commit_index = max(self._commit_index, proposal.log_index)
                logger.debug(f"Raft committed: {proposal_id[:8]}")

    async def finalize(self, proposal_id: str) -> ConsensusResult:
        """
        Finalize consensus on a proposal.

        Returns result once the entry is committed.
        """
        if proposal_id not in self._proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")

        proposal = self._proposals[proposal_id]

        if proposal.result:
            return proposal.result

        n_acks = len(proposal.ack_nodes)

        if proposal.committed:
            result = ConsensusResult(
                proposal_id=proposal_id,
                finalized=True,
                value=proposal.value,
                round_number=proposal.log_index,
                votes_for=n_acks,
                votes_against=0,
                finality_type=self.finality_type,
                confidence=n_acks / self._total_nodes,
                agreeing_nodes=frozenset(proposal.ack_nodes),
                metadata={
                    "term": proposal.term,
                    "log_index": proposal.log_index,
                    "commit_index": self._commit_index,
                },
            )
        else:
            result = ConsensusResult(
                proposal_id=proposal_id,
                finalized=False,
                value=None,
                round_number=proposal.log_index,
                votes_for=n_acks,
                votes_against=0,
                finality_type=self.finality_type,
                confidence=n_acks / self._total_nodes if self._total_nodes > 0 else 0.0,
                metadata={
                    "term": proposal.term,
                    "log_index": proposal.log_index,
                    "majority_needed": self._majority,
                    "current_acks": n_acks,
                },
            )

        proposal.result = result
        proposal.applied = proposal.committed

        return result

    # ───────────────────────────────────────────────────────────────────────────
    #  RAFT-SPECIFIC METHODS
    # ───────────────────────────────────────────────────────────────────────────

    def create_append_entries(
        self,
        follower_id: str,
        entries: Optional[List[LogEntry]] = None,
    ) -> AppendEntriesRequest:
        """
        Create AppendEntries RPC request for a follower.

        This is used for both heartbeats (empty entries) and log replication.
        """
        next_idx = self._next_index.get(follower_id, len(self._log) + 1)
        prev_log_index = next_idx - 1
        prev_log_term = 0

        if prev_log_index > 0 and prev_log_index <= len(self._log):
            prev_log_term = self._log[prev_log_index - 1].term

        if entries is None:
            # Send new entries from next_index
            entries = self._log[next_idx - 1:] if next_idx <= len(self._log) else []

        return AppendEntriesRequest(
            term=self._current_term,
            leader_id=self._node_id,
            prev_log_index=prev_log_index,
            prev_log_term=prev_log_term,
            entries=entries,
            leader_commit=self._commit_index,
        )

    async def handle_append_entries(
        self,
        request: AppendEntriesRequest,
    ) -> AppendEntriesResponse:
        """
        Handle AppendEntries RPC (follower).

        Returns success if log is consistent and entries are appended.
        """
        # Reply false if term < currentTerm
        if request.term < self._current_term:
            return AppendEntriesResponse(
                term=self._current_term,
                success=False,
            )

        # Update term if needed
        if request.term > self._current_term:
            self._current_term = request.term
            self._state = RaftState.FOLLOWER
            self._voted_for = None

        self._leader_id = request.leader_id

        # Reply false if log doesn't contain entry at prevLogIndex
        # matching prevLogTerm
        if request.prev_log_index > 0:
            if request.prev_log_index > len(self._log):
                return AppendEntriesResponse(
                    term=self._current_term,
                    success=False,
                )

            if self._log[request.prev_log_index - 1].term != request.prev_log_term:
                return AppendEntriesResponse(
                    term=self._current_term,
                    success=False,
                )

        # Append new entries (delete conflicting)
        for entry in request.entries:
            if entry.index <= len(self._log):
                if self._log[entry.index - 1].term != entry.term:
                    # Delete conflicting entry and all that follow
                    self._log = self._log[:entry.index - 1]
                    self._log.append(entry)
                # else: entry already exists, skip
            else:
                self._log.append(entry)

        # Update commit index
        if request.leader_commit > self._commit_index:
            last_new_index = request.entries[-1].index if request.entries else 0
            self._commit_index = min(
                request.leader_commit,
                max(last_new_index, len(self._log))
            )

        return AppendEntriesResponse(
            term=self._current_term,
            success=True,
            match_index=len(self._log),
        )

    async def handle_append_response(
        self,
        follower_id: str,
        response: AppendEntriesResponse,
    ) -> None:
        """
        Handle AppendEntries response (leader).

        Updates next_index and match_index for follower.
        """
        if response.term > self._current_term:
            # Step down if we see higher term
            self._current_term = response.term
            self._state = RaftState.FOLLOWER
            self._voted_for = None
            return

        if response.success:
            self._match_index[follower_id] = response.match_index
            self._next_index[follower_id] = response.match_index + 1

            # Check if we can advance commit_index
            await self._try_advance_commit_index()
        else:
            # Decrement next_index and retry
            current = self._next_index.get(follower_id, 1)
            self._next_index[follower_id] = max(1, current - 1)

    async def _try_advance_commit_index(self) -> None:
        """
        Advance commit_index if a majority has replicated.

        From Raft paper: "If there exists an N such that N > commitIndex,
        a majority of matchIndex[i] >= N, and log[N].term == currentTerm:
        set commitIndex = N"
        """
        for n in range(len(self._log), self._commit_index, -1):
            if self._log[n - 1].term != self._current_term:
                continue

            replicated_count = 1  # Leader has it
            for match_idx in self._match_index.values():
                if match_idx >= n:
                    replicated_count += 1

            if replicated_count >= self._majority:
                self._commit_index = n

                # Update proposals that are now committed
                for proposal in self._proposals.values():
                    if proposal.log_index <= n and not proposal.committed:
                        proposal.committed = True
                        logger.debug(f"Raft committed via replication: {proposal.proposal_id[:8]}")

                break

    # ───────────────────────────────────────────────────────────────────────────
    #  LEADER ELECTION
    # ───────────────────────────────────────────────────────────────────────────

    def create_request_vote(self) -> RequestVoteRequest:
        """Create RequestVote RPC for election."""
        last_log_index = len(self._log)
        last_log_term = self._log[-1].term if self._log else 0

        return RequestVoteRequest(
            term=self._current_term,
            candidate_id=self._node_id,
            last_log_index=last_log_index,
            last_log_term=last_log_term,
        )

    async def handle_request_vote(
        self,
        request: RequestVoteRequest,
    ) -> RequestVoteResponse:
        """
        Handle RequestVote RPC.

        Grant vote if:
        1. Candidate's term >= currentTerm
        2. Haven't voted for someone else this term
        3. Candidate's log is at least as up-to-date as ours
        """
        # Reply false if term < currentTerm
        if request.term < self._current_term:
            return RequestVoteResponse(
                term=self._current_term,
                vote_granted=False,
            )

        # Update term if needed
        if request.term > self._current_term:
            self._current_term = request.term
            self._state = RaftState.FOLLOWER
            self._voted_for = None

        # Check if we can grant vote
        vote_granted = False

        if self._voted_for is None or self._voted_for == request.candidate_id:
            # Check if candidate's log is up-to-date
            last_log_index = len(self._log)
            last_log_term = self._log[-1].term if self._log else 0

            log_ok = (
                request.last_log_term > last_log_term or
                (request.last_log_term == last_log_term and
                 request.last_log_index >= last_log_index)
            )

            if log_ok:
                vote_granted = True
                self._voted_for = request.candidate_id

        return RequestVoteResponse(
            term=self._current_term,
            vote_granted=vote_granted,
        )

    async def start_election(self) -> None:
        """
        Start a new election (called on election timeout).

        Transitions to CANDIDATE state and requests votes.
        """
        self._current_term += 1
        self._state = RaftState.CANDIDATE
        self._voted_for = self._node_id  # Vote for self

        logger.info(f"Raft election started: term={self._current_term}")

    async def become_leader(self) -> None:
        """
        Transition to LEADER state after winning election.

        Initializes leader state for all followers.
        """
        self._state = RaftState.LEADER
        self._leader_id = self._node_id

        # Initialize next_index and match_index
        next_idx = len(self._log) + 1
        for i in range(self._total_nodes):
            node_id = f"node_{i}"
            if node_id != self._node_id:
                self._next_index[node_id] = next_idx
                self._match_index[node_id] = 0

        logger.info(f"Raft became leader: term={self._current_term}")

    # ───────────────────────────────────────────────────────────────────────────
    #  STATE ACCESS
    # ───────────────────────────────────────────────────────────────────────────

    async def get_state(self) -> Dict[str, Any]:
        """Get current protocol state."""
        base = await super().get_state()
        base.update({
            "node_id": self._node_id,
            "state": self._state.name,
            "is_leader": self.is_leader,
            "current_term": self._current_term,
            "commit_index": self._commit_index,
            "log_length": len(self._log),
            "majority": self._majority,
            "leader_id": self._leader_id,
            "active_proposals": len([p for p in self._proposals.values() if not p.applied]),
        })
        return base
