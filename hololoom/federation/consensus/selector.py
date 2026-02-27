"""
Consensus Selector - Auto-select algorithm by network size.

Automatically chooses the optimal consensus algorithm based on:
- Network size (number of nodes)
- Fault tolerance requirements (crash-only vs Byzantine)
- Latency requirements (instant vs probabilistic finality)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Type

from .protocol import ConsensusProtocol, FinalityType
from .quorum import QuorumConsensus
from .raft import RaftConsensus
from .pbft import PBFTConsensus
from .avalanche import AvalancheConsensus, AvalancheParams


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  SELECTION CRITERIA
# ═══════════════════════════════════════════════════════════════════════════════


class FaultModel(Enum):
    """Type of faults the network must tolerate."""

    CRASH_ONLY = auto()     # Only crash faults (nodes stop responding)
    BYZANTINE = auto()       # Byzantine faults (nodes may lie/attack)


@dataclass
class SelectionCriteria:
    """
    Criteria for consensus algorithm selection.

    Attributes:
        node_count: Number of nodes in the network
        fault_model: Type of faults to tolerate
        require_instant_finality: Whether instant finality is required
        latency_sensitive: Whether low latency is critical
    """

    node_count: int
    fault_model: FaultModel = FaultModel.CRASH_ONLY
    require_instant_finality: bool = False
    latency_sensitive: bool = False

    def __post_init__(self):
        if self.node_count < 1:
            raise ValueError(f"node_count must be >= 1, got {self.node_count}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════


# Network size thresholds for algorithm selection
RAFT_MAX_NODES = 20        # Raft works well up to ~20 nodes
QUORUM_MAX_NODES = 100     # Quorum works well up to ~100 nodes
PBFT_MAX_NODES = 1000      # PBFT works well up to ~1000 nodes
# Above 1000 nodes: Avalanche


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSENSUS SELECTOR
# ═══════════════════════════════════════════════════════════════════════════════


class ConsensusSelector:
    """
    Selects optimal consensus algorithm based on network conditions.

    Algorithm Selection Matrix:
    ┌────────────┬───────────────┬───────────────────────────────────────┐
    │ Nodes      │ Crash-Only    │ Byzantine Required                    │
    ├────────────┼───────────────┼───────────────────────────────────────┤
    │ <20        │ Raft          │ PBFT (small network, Byzantine)       │
    │ 20-100     │ Quorum        │ PBFT                                  │
    │ 100-1000   │ PBFT          │ PBFT                                  │
    │ 1000+      │ Avalanche     │ Avalanche                             │
    └────────────┴───────────────┴───────────────────────────────────────┘

    Special Cases:
    - Instant finality required + 1000+ nodes: PBFT (slow but guaranteed)
    - Latency sensitive + Byzantine: Consider Quorum with verification
    """

    def select(
        self,
        criteria: SelectionCriteria,
    ) -> tuple[Type[ConsensusProtocol], dict]:
        """
        Select the optimal consensus algorithm.

        Args:
            criteria: Selection criteria

        Returns:
            Tuple of (algorithm class, configuration dict)
        """
        node_count = criteria.node_count
        fault_model = criteria.fault_model
        require_instant = criteria.require_instant_finality

        # 1000+ nodes: Avalanche (unless instant finality required)
        if node_count > PBFT_MAX_NODES:
            if require_instant:
                # Instant finality at scale is expensive but possible with PBFT
                logger.warning(
                    f"PBFT selected for {node_count} nodes with instant finality. "
                    "This may have high message complexity."
                )
                return PBFTConsensus, self._pbft_config(node_count)
            return AvalancheConsensus, self._avalanche_config(node_count)

        # 100-1000 nodes: PBFT
        if node_count > QUORUM_MAX_NODES:
            return PBFTConsensus, self._pbft_config(node_count)

        # 20-100 nodes: Quorum (crash-only) or PBFT (Byzantine)
        if node_count > RAFT_MAX_NODES:
            if fault_model == FaultModel.BYZANTINE:
                return PBFTConsensus, self._pbft_config(node_count)
            return QuorumConsensus, self._quorum_config(node_count)

        # <20 nodes: Raft (crash-only) or PBFT (Byzantine)
        if fault_model == FaultModel.BYZANTINE:
            return PBFTConsensus, self._pbft_config(node_count)
        return RaftConsensus, self._raft_config(node_count)

    def _raft_config(self, node_count: int) -> dict:
        """Generate Raft configuration."""
        return {
            "total_nodes": node_count,
            "election_timeout_range": (0.15, 0.3),
            "heartbeat_interval": 0.05,
        }

    def _quorum_config(self, node_count: int) -> dict:
        """Generate Quorum configuration."""
        # 2/3 threshold for crash fault tolerance
        min_votes = max(3, (2 * node_count) // 3)
        return {
            "threshold": 0.66,
            "min_votes": min_votes,
            "timeout_seconds": 30.0,
        }

    def _pbft_config(self, node_count: int) -> dict:
        """Generate PBFT configuration."""
        return {
            "total_nodes": node_count,
            "timeout_seconds": 30.0,
        }

    def _avalanche_config(self, node_count: int) -> dict:
        """Generate Avalanche configuration."""
        # Scale k with network size (but cap at 50)
        k = min(50, max(20, node_count // 50))
        return {
            "total_nodes": node_count,
            "params": AvalancheParams(
                k=k,
                alpha=0.8,
                beta1=11,
                beta2=20,
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def select_consensus(
    node_count: int,
    *,
    byzantine_tolerant: bool = False,
    require_instant_finality: bool = False,
    latency_sensitive: bool = False,
) -> ConsensusProtocol:
    """
    Select and instantiate the optimal consensus algorithm.

    This is the main entry point for automatic consensus selection.

    Args:
        node_count: Number of nodes in the network
        byzantine_tolerant: Whether Byzantine fault tolerance is needed
        require_instant_finality: Whether instant finality is required
        latency_sensitive: Whether low latency is critical

    Returns:
        Configured ConsensusProtocol instance

    Example:
        # Simple crash-fault tolerant consensus for 10 nodes
        consensus = select_consensus(10)  # Returns RaftConsensus

        # Byzantine tolerant for 500 nodes
        consensus = select_consensus(500, byzantine_tolerant=True)  # Returns PBFTConsensus

        # Large network
        consensus = select_consensus(5000)  # Returns AvalancheConsensus
    """
    criteria = SelectionCriteria(
        node_count=node_count,
        fault_model=FaultModel.BYZANTINE if byzantine_tolerant else FaultModel.CRASH_ONLY,
        require_instant_finality=require_instant_finality,
        latency_sensitive=latency_sensitive,
    )

    selector = ConsensusSelector()
    algorithm_class, config = selector.select(criteria)

    logger.info(
        f"Selected {algorithm_class.__name__} for {node_count} nodes "
        f"(byzantine={byzantine_tolerant}, instant_finality={require_instant_finality})"
    )

    return algorithm_class(**config)


def get_recommended_algorithm(node_count: int) -> str:
    """
    Get the name of the recommended algorithm for a network size.

    Args:
        node_count: Number of nodes

    Returns:
        Algorithm name string
    """
    if node_count > PBFT_MAX_NODES:
        return "avalanche"
    elif node_count > QUORUM_MAX_NODES:
        return "pbft"
    elif node_count > RAFT_MAX_NODES:
        return "quorum"
    else:
        return "raft"


def describe_algorithm(name: str) -> dict:
    """
    Get description and characteristics of a consensus algorithm.

    Args:
        name: Algorithm name (raft, quorum, pbft, avalanche)

    Returns:
        Dict with algorithm characteristics
    """
    descriptions = {
        "raft": {
            "name": "Raft",
            "description": "Leader-based log replication for crash fault tolerance",
            "best_for": "Small networks (<20 nodes)",
            "fault_tolerance": "Crash-only (f < n/2)",
            "finality": "Instant",
            "message_complexity": "O(n)",
        },
        "quorum": {
            "name": "Quorum",
            "description": "Simple threshold-based voting",
            "best_for": "Small to medium networks (20-100 nodes)",
            "fault_tolerance": "Crash-only (f < n/3)",
            "finality": "Instant",
            "message_complexity": "O(n)",
        },
        "pbft": {
            "name": "PBFT",
            "description": "Practical Byzantine Fault Tolerance with three-phase protocol",
            "best_for": "Medium networks (100-1000 nodes)",
            "fault_tolerance": "Byzantine (f < n/3)",
            "finality": "Instant",
            "message_complexity": "O(n²)",
        },
        "avalanche": {
            "name": "Avalanche",
            "description": "Probabilistic sampling for large-scale consensus",
            "best_for": "Large networks (1000+ nodes)",
            "fault_tolerance": "Byzantine (f < n/4)",
            "finality": "Probabilistic",
            "message_complexity": "O(k log n)",
        },
    }

    return descriptions.get(name.lower(), {"error": f"Unknown algorithm: {name}"})
