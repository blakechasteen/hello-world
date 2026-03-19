"""
HoloLoom Federation Consensus - Pluggable consensus algorithms.

Provides 4 consensus algorithms for different network scales:
- Raft: Simple, leader-based (<20 nodes)
- Quorum: Threshold voting (20-100 nodes)
- PBFT: Byzantine fault tolerant (100-1000 nodes)
- Avalanche: Probabilistic sampling (1000+ nodes)

Quick Start:
    # Auto-select based on network size
    from hololoom.federation.consensus import select_consensus

    consensus = select_consensus(500, byzantine_tolerant=True)
    proposal_id = await consensus.propose(my_value)
    await consensus.vote(proposal_id, vote=True, node_id="node_1")
    result = await consensus.finalize(proposal_id)

    if result.finalized:
        print(f"Consensus reached: {result.value}")

Algorithm Selection:
    | Nodes    | Crash-Only | Byzantine  |
    |----------|------------|------------|
    | <20      | Raft       | PBFT       |
    | 20-100   | Quorum     | PBFT       |
    | 100-1000 | PBFT       | PBFT       |
    | 1000+    | Avalanche  | Avalanche  |
"""

from .avalanche import (
    AvalancheConsensus,
    AvalancheParams,
    AvalanchePreference,
    PreferenceState,
)
from .pbft import (
    CommitMessage,
    PBFTConsensus,
    PBFTMessage,
    PBFTPhase,
    PrepareMessage,
    PrePrepareMessage,
)
from .protocol import (
    # Core protocol
    ConsensusProtocol,
    ConsensusResult,
    FinalityType,
    # Response-specific
    ResponseConsensusProtocol,
    ResponseConsensusResult,
)
from .quorum import (
    QuorumConsensus,
    ResponseQuorumConsensus,
)
from .raft import (
    AppendEntriesRequest,
    AppendEntriesResponse,
    LogEntry,
    RaftConsensus,
    RaftState,
    RequestVoteRequest,
    RequestVoteResponse,
)
from .selector import (
    ConsensusSelector,
    FaultModel,
    SelectionCriteria,
    describe_algorithm,
    get_recommended_algorithm,
    select_consensus,
)

__all__ = [
    # Protocol ABC
    "ConsensusProtocol",
    "ConsensusResult",
    "FinalityType",
    "ResponseConsensusProtocol",
    "ResponseConsensusResult",
    # Quorum
    "QuorumConsensus",
    "ResponseQuorumConsensus",
    # Raft
    "RaftConsensus",
    "RaftState",
    "LogEntry",
    "AppendEntriesRequest",
    "AppendEntriesResponse",
    "RequestVoteRequest",
    "RequestVoteResponse",
    # PBFT
    "PBFTConsensus",
    "PBFTPhase",
    "PBFTMessage",
    "PrePrepareMessage",
    "PrepareMessage",
    "CommitMessage",
    # Avalanche
    "AvalancheConsensus",
    "AvalancheParams",
    "AvalanchePreference",
    "PreferenceState",
    # Selector
    "ConsensusSelector",
    "SelectionCriteria",
    "FaultModel",
    "select_consensus",
    "get_recommended_algorithm",
    "describe_algorithm",
]
