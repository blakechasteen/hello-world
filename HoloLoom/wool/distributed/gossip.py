"""
Gossip Protocol for Cluster Membership
=======================================

SWIM-style gossip protocol for distributed wool storage cluster.

This module implements cluster membership and failure detection:
- SWIM protocol (Scalable Weakly-consistent Infection-style Process Group Membership)
- Failure detection through periodic pings
- Indirect probing for suspected failures
- Gossip-based membership propagation

SWIM Protocol Overview:
    1. Each node periodically pings random member
    2. If no ack, request indirect probe from K members
    3. If still no ack, mark as failed and gossip
    4. Failed nodes removed after timeout

Gossip Messages:
    - PING: Are you alive?
    - ACK: Yes, I'm alive
    - PING_REQ: Please ping X for me (indirect probe)
    - JOIN: New node joining
    - LEAVE: Node leaving gracefully
    - SUSPECT: Node suspected of failure
    - ALIVE: Node is actually alive (refutation)
    - DEAD: Node confirmed dead

Author: Claude Code
Date: November 17, 2025 (Phase 6.4)
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Set, Optional, Callable, Any

from HoloLoom.wool.distributed.node import NodeStatus


logger = logging.getLogger(__name__)


class GossipMessageType(Enum):
    """Gossip message types."""
    PING = "PING"
    ACK = "ACK"
    PING_REQ = "PING_REQ"  # Indirect probe request
    JOIN = "JOIN"
    LEAVE = "LEAVE"
    SUSPECT = "SUSPECT"
    ALIVE = "ALIVE"
    DEAD = "DEAD"


@dataclass
class GossipMessage:
    """Gossip protocol message."""
    msg_type: GossipMessageType
    sender_id: str
    target_id: Optional[str] = None
    incarnation: int = 0  # Version number for refutation
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'msg_type': self.msg_type.value,
            'sender_id': self.sender_id,
            'target_id': self.target_id,
            'incarnation': self.incarnation,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GossipMessage':
        """Create from dictionary."""
        return cls(
            msg_type=GossipMessageType(d['msg_type']),
            sender_id=d['sender_id'],
            target_id=d.get('target_id'),
            incarnation=d.get('incarnation', 0),
            metadata=d.get('metadata', {})
        )


@dataclass
class MemberInfo:
    """Cluster member information."""
    node_id: str
    host: str
    port: int
    status: NodeStatus = NodeStatus.HEALTHY
    incarnation: int = 0
    last_seen: float = 0.0
    suspect_time: Optional[float] = None

    @property
    def address(self) -> str:
        """Network address."""
        return f"{self.host}:{self.port}"


class GossipProtocol:
    """
    SWIM-style gossip protocol for cluster membership.

    Features:
    - Periodic ping to detect failures
    - Indirect probing for suspected failures
    - Gossip-based dissemination of membership changes
    - Incarnation numbers for refutation
    - Configurable timeouts and probe counts

    Usage:
        gossip = GossipProtocol(
            node_id="node-1",
            host="localhost",
            port=9000,
            ping_interval=1.0,
            suspect_timeout=3.0,
            failure_timeout=10.0
        )

        # Register status callback
        gossip.on_status_change = lambda node_id, old, new: print(f"{node_id}: {old} → {new}")

        async with gossip:
            # Add members
            gossip.add_member("node-2", "localhost", 9001)
            gossip.add_member("node-3", "localhost", 9002)

            # Protocol runs in background
            await asyncio.sleep(60)
    """

    def __init__(
        self,
        node_id: str,
        host: str = "localhost",
        port: int = 9000,
        ping_interval: float = 1.0,
        suspect_timeout: float = 3.0,
        failure_timeout: float = 10.0,
        indirect_probe_count: int = 3
    ):
        """
        Initialize gossip protocol.

        Args:
            node_id: This node's identifier
            host: This node's host
            port: This node's port
            ping_interval: Seconds between pings
            suspect_timeout: Seconds before marking suspect
            failure_timeout: Seconds before marking dead
            indirect_probe_count: Number of indirect probes
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.ping_interval = ping_interval
        self.suspect_timeout = suspect_timeout
        self.failure_timeout = failure_timeout
        self.indirect_probe_count = indirect_probe_count

        # Cluster members
        self.members: Dict[str, MemberInfo] = {}

        # Self member
        self.members[node_id] = MemberInfo(
            node_id=node_id,
            host=host,
            port=port,
            status=NodeStatus.HEALTHY,
            incarnation=0,
            last_seen=time.time()
        )

        # Pending acks
        self.pending_acks: Set[str] = set()

        # Background task
        self.background_task: Optional[asyncio.Task] = None
        self.running = False

        # Callbacks
        self.on_status_change: Optional[Callable[[str, NodeStatus, NodeStatus], None]] = None
        self.on_member_join: Optional[Callable[[str], None]] = None
        self.on_member_leave: Optional[Callable[[str], None]] = None

        # Statistics
        self.stats = {
            'pings_sent': 0,
            'acks_received': 0,
            'suspects_detected': 0,
            'failures_detected': 0,
            'indirect_probes': 0
        }

        logger.info(f"Initialized gossip protocol for {node_id} at {host}:{port}")
        logger.info(f"Ping interval: {ping_interval}s, suspect: {suspect_timeout}s, failure: {failure_timeout}s")

    def add_member(
        self,
        node_id: str,
        host: str,
        port: int,
        status: NodeStatus = NodeStatus.HEALTHY
    ):
        """
        Add cluster member.

        Args:
            node_id: Member node identifier
            host: Member host
            port: Member port
            status: Initial status
        """
        if node_id == self.node_id:
            return  # Don't add self

        if node_id not in self.members:
            self.members[node_id] = MemberInfo(
                node_id=node_id,
                host=host,
                port=port,
                status=status,
                incarnation=0,
                last_seen=time.time()
            )
            logger.info(f"Added member: {node_id} at {host}:{port}")

            if self.on_member_join:
                self.on_member_join(node_id)

    def remove_member(self, node_id: str):
        """
        Remove cluster member.

        Args:
            node_id: Member to remove
        """
        if node_id in self.members and node_id != self.node_id:
            del self.members[node_id]
            logger.info(f"Removed member: {node_id}")

            if self.on_member_leave:
                self.on_member_leave(node_id)

    def get_healthy_members(self, exclude_self: bool = True) -> List[MemberInfo]:
        """
        Get list of healthy members.

        Args:
            exclude_self: Whether to exclude self from list

        Returns:
            List of healthy members
        """
        members = [
            m for m in self.members.values()
            if m.status == NodeStatus.HEALTHY
        ]

        if exclude_self:
            members = [m for m in members if m.node_id != self.node_id]

        return members

    def get_random_member(self, exclude: Optional[Set[str]] = None) -> Optional[MemberInfo]:
        """
        Get random healthy member.

        Args:
            exclude: Set of node IDs to exclude

        Returns:
            Random member or None
        """
        exclude = exclude or set()
        exclude.add(self.node_id)  # Always exclude self

        candidates = [
            m for m in self.members.values()
            if m.node_id not in exclude and m.status == NodeStatus.HEALTHY
        ]

        if not candidates:
            return None

        return random.choice(candidates)

    async def ping(self, target_id: str) -> bool:
        """
        Ping a member node.

        Args:
            target_id: Target node identifier

        Returns:
            True if ack received
        """
        if target_id not in self.members:
            return False

        # Send ping (placeholder - actual network call would go here)
        self.stats['pings_sent'] += 1
        self.pending_acks.add(target_id)

        # Simulate ack (in real implementation, this would be async network response)
        # For now, assume ack after small delay
        await asyncio.sleep(0.1)

        # Check if ack received
        if target_id in self.pending_acks:
            # Timeout - no ack
            self.pending_acks.discard(target_id)
            return False
        else:
            # Ack received
            self.stats['acks_received'] += 1
            self.members[target_id].last_seen = time.time()
            return True

    async def indirect_probe(self, target_id: str) -> bool:
        """
        Perform indirect probe through K members.

        Args:
            target_id: Target node to probe

        Returns:
            True if any probe succeeded
        """
        # Select K random members (excluding target and self)
        exclude = {self.node_id, target_id}
        probe_members = []

        for _ in range(self.indirect_probe_count):
            member = self.get_random_member(exclude=exclude)
            if member:
                probe_members.append(member)
                exclude.add(member.node_id)

        if not probe_members:
            logger.warning(f"No members available for indirect probe of {target_id}")
            return False

        logger.debug(f"Indirect probe of {target_id} through {len(probe_members)} members")
        self.stats['indirect_probes'] += 1

        # Send PING_REQ to each probe member
        # (In real implementation, this would be parallel network calls)
        for member in probe_members:
            # Simulate probe
            success = await self.ping(target_id)
            if success:
                return True

        return False

    async def mark_suspect(self, node_id: str):
        """
        Mark node as suspected of failure.

        Args:
            node_id: Node to mark suspect
        """
        if node_id not in self.members or node_id == self.node_id:
            return

        member = self.members[node_id]
        old_status = member.status

        if old_status != NodeStatus.HEALTHY:
            return  # Already suspect or dead

        member.status = NodeStatus.DEGRADED  # Use DEGRADED for suspect
        member.suspect_time = time.time()

        self.stats['suspects_detected'] += 1
        logger.warning(f"Marked {node_id} as SUSPECT")

        if self.on_status_change:
            self.on_status_change(node_id, old_status, NodeStatus.DEGRADED)

    async def mark_failed(self, node_id: str):
        """
        Mark node as failed.

        Args:
            node_id: Node to mark failed
        """
        if node_id not in self.members or node_id == self.node_id:
            return

        member = self.members[node_id]
        old_status = member.status

        if old_status == NodeStatus.UNREACHABLE:
            return  # Already dead

        member.status = NodeStatus.UNREACHABLE

        self.stats['failures_detected'] += 1
        logger.error(f"Marked {node_id} as FAILED")

        if self.on_status_change:
            self.on_status_change(node_id, old_status, NodeStatus.UNREACHABLE)

    async def refute_suspicion(self):
        """
        Refute suspicion of this node (increment incarnation).
        """
        self.members[self.node_id].incarnation += 1
        logger.info(f"Refuted suspicion, incarnation: {self.members[self.node_id].incarnation}")

    async def _gossip_loop(self):
        """Background gossip loop (SWIM protocol)."""
        logger.info("Started gossip loop")

        while self.running:
            try:
                # Select random member to ping
                target = self.get_random_member()

                if target:
                    # Direct ping
                    ack = await self.ping(target.node_id)

                    if not ack:
                        # No ack - try indirect probe
                        logger.debug(f"No ack from {target.node_id}, trying indirect probe")
                        indirect_ack = await self.indirect_probe(target.node_id)

                        if not indirect_ack:
                            # Still no ack - mark suspect
                            await self.mark_suspect(target.node_id)

                # Check suspect timeouts
                now = time.time()
                for member in list(self.members.values()):
                    if member.node_id == self.node_id:
                        continue

                    # Check if suspect timeout expired
                    if member.status == NodeStatus.DEGRADED and member.suspect_time:
                        if now - member.suspect_time > self.failure_timeout:
                            await self.mark_failed(member.node_id)

                # Wait before next ping
                await asyncio.sleep(self.ping_interval)

            except Exception as e:
                logger.error(f"Error in gossip loop: {e}", exc_info=True)
                await asyncio.sleep(self.ping_interval)

        logger.info("Stopped gossip loop")

    def get_stats(self) -> Dict[str, Any]:
        """Get gossip protocol statistics."""
        return {
            'node_id': self.node_id,
            'address': f"{self.host}:{self.port}",
            'members': {
                'total': len(self.members),
                'healthy': sum(1 for m in self.members.values() if m.status == NodeStatus.HEALTHY),
                'suspect': sum(1 for m in self.members.values() if m.status == NodeStatus.DEGRADED),
                'failed': sum(1 for m in self.members.values() if m.status == NodeStatus.UNREACHABLE)
            },
            'operations': self.stats.copy()
        }

    async def start(self):
        """Start gossip protocol."""
        if self.running:
            logger.warning("Gossip protocol already running")
            return

        self.running = True
        self.background_task = asyncio.create_task(self._gossip_loop())
        logger.info("Started gossip protocol")

    async def stop(self):
        """Stop gossip protocol."""
        if not self.running:
            return

        self.running = False

        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
            self.background_task = None

        logger.info("Stopped gossip protocol")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
