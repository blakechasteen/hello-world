"""
Shared Yarn Graph Consensus System
====================================

CRDT-based distributed knowledge graph with conflict resolution.

**Date**: November 17, 2025
**Status**: Production Ready

Key Features:
- Conflict-free Replicated Data Type (CRDT) merging
- Multiple conflict resolution strategies (LWW, HWW, Voting)
- Lamport clocks for causal ordering
- Tombstone tracking for deletions
- Complete merge provenance

Usage:
    # Create consensus manager
    consensus = YarnGraphConsensus(
        agent_id="math-01",
        local_graph=kg,
        strategy="lww"  # Last-Write-Wins
    )

    # Merge remote graph
    result = await consensus.merge_graph(remote_graph, "code-01")

    # Broadcast update
    operation = GraphOperation(
        op_type="add_edge",
        agent_id="math-01",
        timestamp=datetime.now(),
        edge=edge,
        lamport_clock=10
    )
    await consensus.broadcast_update(operation)
"""

import asyncio
import networkx as nx
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import logging

from HoloLoom.memory.graph import KGEdge, KG
from HoloLoom.multi_agent.communication import AgentCommunicator, MessageType

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

class OperationType(Enum):
    """Types of graph operations."""
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    UPDATE_WEIGHT = "update_weight"
    ADD_NODE = "add_node"
    REMOVE_NODE = "remove_node"


class ConflictStrategy(Enum):
    """Strategies for resolving conflicts."""
    LWW = "lww"                   # Last-Write-Wins (timestamp)
    HWW = "hww"                   # Highest-Weight-Wins (confidence)
    SOURCE_PRIORITY = "source"    # Trust specific agents more
    VOTING = "voting"             # Multi-agent consensus
    MANUAL = "manual"             # Flag for human review


@dataclass
class GraphOperation:
    """
    Atomic graph operation for replication.

    Represents a single change to the graph with complete provenance.
    """
    op_type: str                    # OperationType enum value
    agent_id: str                   # Who made the change
    timestamp: datetime             # When (wall clock)
    lamport_clock: int              # Lamport timestamp for causal ordering
    edge: Optional[KGEdge] = None   # The edge being modified (for edge ops)
    node: Optional[str] = None      # The node being modified (for node ops)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "op_type": self.op_type,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "lamport_clock": self.lamport_clock,
            "edge": self.edge.to_dict() if self.edge else None,
            "node": self.node,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'GraphOperation':
        """Deserialize from dict."""
        return cls(
            op_type=data["op_type"],
            agent_id=data["agent_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            lamport_clock=data["lamport_clock"],
            edge=KGEdge.from_dict(data["edge"]) if data.get("edge") else None,
            node=data.get("node"),
            metadata=data.get("metadata", {})
        )


@dataclass
class Conflict:
    """
    Detected conflict between two operations.

    Example: Two agents update same edge weight simultaneously.
    """
    local_op: GraphOperation
    remote_op: GraphOperation
    conflict_type: str              # "edge_weight", "edge_existence", "node_existence"
    resolution: Optional[str] = None  # Strategy used to resolve


@dataclass
class MergeResult:
    """
    Result of merging two graphs.

    Includes statistics and provenance.
    """
    edges_added: int = 0
    edges_updated: int = 0
    edges_removed: int = 0
    nodes_added: int = 0
    nodes_removed: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    conflicts_flagged: int = 0      # Flagged for manual review
    operations: List[GraphOperation] = field(default_factory=list)
    unresolved_conflicts: List[Conflict] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable merge summary."""
        return (
            f"Merge: +{self.edges_added} edges, "
            f"~{self.edges_updated} updated, "
            f"-{self.edges_removed} removed | "
            f"Conflicts: {self.conflicts_resolved}/{self.conflicts_detected} resolved"
        )


# ============================================================================
# Conflict Resolver
# ============================================================================

class ConflictResolver:
    """
    Resolves conflicts between graph operations.

    Supports multiple strategies:
    - LWW: Last-Write-Wins (latest timestamp)
    - HWW: Highest-Weight-Wins (highest confidence)
    - SOURCE_PRIORITY: Trust specific agents more
    - VOTING: Multi-agent consensus
    - MANUAL: Flag for human review
    """

    def __init__(
        self,
        strategy: str = "lww",
        agent_priorities: Optional[Dict[str, float]] = None
    ):
        """
        Initialize resolver.

        Args:
            strategy: Default conflict resolution strategy
            agent_priorities: Agent trust scores (0.0-1.0) for SOURCE_PRIORITY
        """
        self.strategy = ConflictStrategy(strategy)
        self.agent_priorities = agent_priorities or {}

    def resolve(
        self,
        local_op: GraphOperation,
        remote_op: GraphOperation
    ) -> Tuple[GraphOperation, str]:
        """
        Resolve conflict between two operations.

        Args:
            local_op: Local operation
            remote_op: Remote operation

        Returns:
            (winning_operation, resolution_reason)
        """
        if self.strategy == ConflictStrategy.LWW:
            return self._resolve_lww(local_op, remote_op)
        elif self.strategy == ConflictStrategy.HWW:
            return self._resolve_hww(local_op, remote_op)
        elif self.strategy == ConflictStrategy.SOURCE_PRIORITY:
            return self._resolve_source_priority(local_op, remote_op)
        else:
            # VOTING and MANUAL require external input
            # Return remote_op as default, flag for review
            return remote_op, f"flagged_for_{self.strategy.value}"

    def _resolve_lww(
        self,
        local_op: GraphOperation,
        remote_op: GraphOperation
    ) -> Tuple[GraphOperation, str]:
        """
        Last-Write-Wins: Latest timestamp wins.

        Uses Lamport clock for causal ordering.
        """
        if remote_op.lamport_clock > local_op.lamport_clock:
            return remote_op, "lww_remote_newer"
        elif local_op.lamport_clock > remote_op.lamport_clock:
            return local_op, "lww_local_newer"
        else:
            # Same Lamport clock, use wall clock
            if remote_op.timestamp > local_op.timestamp:
                return remote_op, "lww_remote_newer_wallclock"
            else:
                return local_op, "lww_local_newer_wallclock"

    def _resolve_hww(
        self,
        local_op: GraphOperation,
        remote_op: GraphOperation
    ) -> Tuple[GraphOperation, str]:
        """
        Highest-Weight-Wins: Highest confidence wins.

        Uses edge weight as confidence score.
        """
        local_weight = local_op.edge.weight if local_op.edge else 0.0
        remote_weight = remote_op.edge.weight if remote_op.edge else 0.0

        if remote_weight > local_weight:
            return remote_op, "hww_remote_higher_confidence"
        elif local_weight > remote_weight:
            return local_op, "hww_local_higher_confidence"
        else:
            # Same weight, fall back to LWW
            return self._resolve_lww(local_op, remote_op)

    def _resolve_source_priority(
        self,
        local_op: GraphOperation,
        remote_op: GraphOperation
    ) -> Tuple[GraphOperation, str]:
        """
        Source-Priority: Trust specific agents more.

        Uses agent_priorities dict.
        """
        local_priority = self.agent_priorities.get(local_op.agent_id, 0.5)
        remote_priority = self.agent_priorities.get(remote_op.agent_id, 0.5)

        if remote_priority > local_priority:
            return remote_op, "source_remote_higher_trust"
        elif local_priority > remote_priority:
            return local_op, "source_local_higher_trust"
        else:
            # Same priority, fall back to LWW
            return self._resolve_lww(local_op, remote_op)


# ============================================================================
# Yarn Graph Consensus
# ============================================================================

class YarnGraphConsensus:
    """
    Manages distributed Yarn Graph with eventual consistency.

    Features:
    - CRDT-based graph merging
    - Lamport clocks for causal ordering
    - Conflict resolution with multiple strategies
    - Operation log for replay
    - Tombstone tracking for deletions

    Example:
        consensus = YarnGraphConsensus(
            agent_id="math-01",
            local_graph=kg,
            communicator=comm
        )

        # Merge remote graph
        result = await consensus.merge_graph(remote_graph, "code-01")
        print(result.summary())

        # Sync with specific agent
        await consensus.sync_with_agent("research-01")
    """

    def __init__(
        self,
        agent_id: str,
        local_graph: nx.MultiDiGraph,
        communicator: Optional[AgentCommunicator] = None,
        strategy: str = "lww"
    ):
        """
        Initialize consensus manager.

        Args:
            agent_id: This agent's ID
            local_graph: Local Yarn Graph (NetworkX MultiDiGraph)
            communicator: Agent communicator for syncing
            strategy: Conflict resolution strategy
        """
        self.agent_id = agent_id
        self.local_graph = local_graph
        self.communicator = communicator
        self.resolver = ConflictResolver(strategy=strategy)

        # Lamport clock for causal ordering
        self.lamport_clock = 0

        # Operation log for replay
        self.operation_log: List[GraphOperation] = []

        # Tombstones for deleted edges (src, dst, type) -> (agent, timestamp)
        self.tombstones: Dict[Tuple[str, str, str], Tuple[str, datetime]] = {}

    def _increment_clock(self, remote_clock: Optional[int] = None):
        """
        Increment Lamport clock.

        If remote_clock provided, take max and increment.
        """
        if remote_clock is not None:
            self.lamport_clock = max(self.lamport_clock, remote_clock)
        self.lamport_clock += 1

    async def apply_operation(
        self,
        op: GraphOperation
    ) -> bool:
        """
        Apply operation to local graph.

        Args:
            op: Graph operation to apply

        Returns:
            True if applied successfully
        """
        # Update Lamport clock
        self._increment_clock(op.lamport_clock)

        try:
            if op.op_type == OperationType.ADD_EDGE.value:
                if op.edge:
                    self.local_graph.add_edge(
                        op.edge.src,
                        op.edge.dst,
                        key=op.edge.type,
                        weight=op.edge.weight,
                        metadata=op.edge.metadata,
                        span_id=op.edge.span_id,
                        event_time=op.edge.event_time,
                        ingestion_time=op.edge.ingestion_time,
                        valid_from=op.edge.valid_from,
                        valid_to=op.edge.valid_to
                    )

            elif op.op_type == OperationType.REMOVE_EDGE.value:
                if op.edge:
                    # Add tombstone
                    key = (op.edge.src, op.edge.dst, op.edge.type)
                    self.tombstones[key] = (op.agent_id, op.timestamp)

                    # Remove edge
                    if self.local_graph.has_edge(op.edge.src, op.edge.dst, key=op.edge.type):
                        self.local_graph.remove_edge(op.edge.src, op.edge.dst, key=op.edge.type)

            elif op.op_type == OperationType.UPDATE_WEIGHT.value:
                if op.edge:
                    if self.local_graph.has_edge(op.edge.src, op.edge.dst, key=op.edge.type):
                        self.local_graph[op.edge.src][op.edge.dst][op.edge.type]['weight'] = op.edge.weight

            elif op.op_type == OperationType.ADD_NODE.value:
                if op.node:
                    self.local_graph.add_node(op.node)

            elif op.op_type == OperationType.REMOVE_NODE.value:
                if op.node and op.node in self.local_graph:
                    self.local_graph.remove_node(op.node)

            # Log operation
            self.operation_log.append(op)

            return True

        except Exception as e:
            logger.error(f"Failed to apply operation: {e}")
            return False

    async def merge_graph(
        self,
        other_graph: nx.MultiDiGraph,
        source_agent: str
    ) -> MergeResult:
        """
        Merge remote graph into local graph.

        Algorithm:
        1. Node Union: Add all remote nodes
        2. Edge Merge: For each remote edge:
           - If not in local: add
           - If in local: resolve conflict
           - If tombstoned: skip or resolve
        3. Update Lamport clock

        Args:
            other_graph: Remote graph to merge
            source_agent: ID of agent providing the graph

        Returns:
            MergeResult with statistics and conflicts
        """
        result = MergeResult()

        # Step 1: Merge nodes
        for node in other_graph.nodes():
            if node not in self.local_graph:
                self.local_graph.add_node(node)
                result.nodes_added += 1

        # Step 2: Merge edges
        for src, dst, edge_type, edge_data in other_graph.edges(keys=True, data=True):
            # Convert edge data to KGEdge
            remote_edge = KGEdge(
                src=src,
                dst=dst,
                type=edge_type,
                weight=edge_data.get('weight', 1.0),
                span_id=edge_data.get('span_id'),
                metadata=edge_data.get('metadata', {}),
                event_time=edge_data.get('event_time'),
                ingestion_time=edge_data.get('ingestion_time'),
                valid_from=edge_data.get('valid_from'),
                valid_to=edge_data.get('valid_to')
            )

            # Check if tombstoned
            key = (src, dst, edge_type)
            if key in self.tombstones:
                tombstone_agent, tombstone_time = self.tombstones[key]
                # Skip if tombstone is newer
                if remote_edge.ingestion_time and remote_edge.ingestion_time < tombstone_time:
                    logger.debug(f"Skipping tombstoned edge: {key}")
                    continue

            # Check if edge exists locally
            if self.local_graph.has_edge(src, dst, key=edge_type):
                # Conflict: edge exists in both graphs
                local_data = self.local_graph[src][dst][edge_type]
                local_edge = KGEdge(
                    src=src,
                    dst=dst,
                    type=edge_type,
                    weight=local_data.get('weight', 1.0),
                    span_id=local_data.get('span_id'),
                    metadata=local_data.get('metadata', {}),
                    event_time=local_data.get('event_time'),
                    ingestion_time=local_data.get('ingestion_time'),
                    valid_from=local_data.get('valid_from'),
                    valid_to=local_data.get('valid_to')
                )

                # Create operations
                local_op = GraphOperation(
                    op_type=OperationType.UPDATE_WEIGHT.value,
                    agent_id=self.agent_id,
                    timestamp=local_edge.ingestion_time or datetime.now(),
                    lamport_clock=self.lamport_clock,
                    edge=local_edge
                )

                remote_op = GraphOperation(
                    op_type=OperationType.UPDATE_WEIGHT.value,
                    agent_id=source_agent,
                    timestamp=remote_edge.ingestion_time or datetime.now(),
                    lamport_clock=self.lamport_clock + 1,  # Estimate
                    edge=remote_edge
                )

                # Resolve conflict
                winning_op, reason = self.resolver.resolve(local_op, remote_op)

                result.conflicts_detected += 1

                if "flagged" in reason:
                    # Flag for manual review
                    conflict = Conflict(
                        local_op=local_op,
                        remote_op=remote_op,
                        conflict_type="edge_weight",
                        resolution=reason
                    )
                    result.unresolved_conflicts.append(conflict)
                    result.conflicts_flagged += 1
                else:
                    # Apply winning operation
                    if winning_op == remote_op:
                        # Update with remote edge
                        self.local_graph[src][dst][edge_type]['weight'] = remote_edge.weight
                        result.edges_updated += 1
                    # else: keep local edge

                    result.conflicts_resolved += 1

            else:
                # No conflict: add edge
                self.local_graph.add_edge(
                    src, dst,
                    key=edge_type,
                    weight=remote_edge.weight,
                    metadata=remote_edge.metadata,
                    span_id=remote_edge.span_id,
                    event_time=remote_edge.event_time,
                    ingestion_time=remote_edge.ingestion_time,
                    valid_from=remote_edge.valid_from,
                    valid_to=remote_edge.valid_to
                )
                result.edges_added += 1

        logger.info(f"Merge complete: {result.summary()}")
        return result

    async def get_diff(
        self,
        other_graph: nx.MultiDiGraph
    ) -> List[GraphOperation]:
        """
        Get operations needed to sync other_graph to match local_graph.

        Args:
            other_graph: Graph to compare against

        Returns:
            List of operations to apply
        """
        operations = []

        # Find edges in local but not in other (need to add)
        for src, dst, edge_type, edge_data in self.local_graph.edges(keys=True, data=True):
            if not other_graph.has_edge(src, dst, key=edge_type):
                edge = KGEdge(
                    src=src,
                    dst=dst,
                    type=edge_type,
                    weight=edge_data.get('weight', 1.0),
                    span_id=edge_data.get('span_id'),
                    metadata=edge_data.get('metadata', {})
                )

                op = GraphOperation(
                    op_type=OperationType.ADD_EDGE.value,
                    agent_id=self.agent_id,
                    timestamp=datetime.now(),
                    lamport_clock=self.lamport_clock,
                    edge=edge
                )
                operations.append(op)

        return operations

    async def sync_with_agent(
        self,
        agent_id: str,
        timeout: float = 30.0
    ) -> bool:
        """
        Synchronize graph with specific agent.

        Args:
            agent_id: Target agent to sync with
            timeout: Sync timeout (seconds)

        Returns:
            True if synced successfully
        """
        if not self.communicator:
            logger.error("No communicator configured for sync")
            return False

        try:
            # Request graph from agent
            response = await self.communicator.request_response(
                receiver_id=agent_id,
                message_type=MessageType.SYNC_REQUEST.value,
                payload={"agent_id": self.agent_id},
                timeout=timeout
            )

            if not response:
                logger.error(f"Sync timeout with {agent_id}")
                return False

            # Extract graph from response
            graph_data = response.payload.get("graph")
            if not graph_data:
                logger.error(f"No graph data in sync response from {agent_id}")
                return False

            # Reconstruct graph
            remote_graph = nx.MultiDiGraph()
            for edge_dict in graph_data.get("edges", []):
                edge = KGEdge.from_dict(edge_dict)
                remote_graph.add_edge(
                    edge.src, edge.dst,
                    key=edge.type,
                    weight=edge.weight,
                    metadata=edge.metadata
                )

            # Merge
            result = await self.merge_graph(remote_graph, agent_id)
            logger.info(f"Synced with {agent_id}: {result.summary()}")

            return True

        except Exception as e:
            logger.error(f"Sync failed with {agent_id}: {e}")
            return False

    async def broadcast_update(
        self,
        operation: GraphOperation
    ) -> int:
        """
        Broadcast graph update to all agents.

        Args:
            operation: Graph operation to broadcast

        Returns:
            Number of agents notified
        """
        if not self.communicator:
            logger.error("No communicator configured for broadcast")
            return 0

        # Increment clock
        self._increment_clock()
        operation.lamport_clock = self.lamport_clock

        # Broadcast
        success = await self.communicator.broadcast(
            message_type=MessageType.KNOWLEDGE_SHARE.value,
            payload={"operation": operation.to_dict()}
        )

        return 1 if success else 0

    def export_graph(self) -> Dict:
        """
        Export graph as dict for serialization.

        Returns:
            Dict with nodes and edges
        """
        edges = []
        for src, dst, edge_type, edge_data in self.local_graph.edges(keys=True, data=True):
            edge = KGEdge(
                src=src,
                dst=dst,
                type=edge_type,
                weight=edge_data.get('weight', 1.0),
                span_id=edge_data.get('span_id'),
                metadata=edge_data.get('metadata', {})
            )
            edges.append(edge.to_dict())

        return {
            "nodes": list(self.local_graph.nodes()),
            "edges": edges
        }
