"""
Gossip Membership - SWIM protocol for decentralized membership.

SWIM = Scalable Weakly-consistent Infection-style Membership

Key properties:
- O(1) message complexity per membership change
- Sublinear failure detection time
- No single point of failure
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from .protocols import MembershipProtocol
from .types import FederationNode, NodeStatus
from .transport import (
    BaseTransport,
    MessageType as TransportMessageType,
    TransportMessage,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  MESSAGE TYPES
# ═══════════════════════════════════════════════════════════════════════════


class MessageType(Enum):
    """SWIM message types."""

    PING = auto()        # Direct probe
    PING_REQ = auto()    # Indirect probe request
    ACK = auto()         # Probe response
    ALIVE = auto()       # Node is alive (override suspect)
    SUSPECT = auto()     # Node may be dead
    DEAD = auto()        # Node confirmed dead
    JOIN = auto()        # Node joining
    LEAVE = auto()       # Node leaving


@dataclass(frozen=True, slots=True)
class GossipMessage:
    """A gossip message."""

    type: MessageType
    sender: str                               # Node ID
    target: str                               # Target node ID
    incarnation: int = 0                      # Lamport-style counter
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════════════════
#  MEMBER STATE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class MemberState:
    """State of a member node."""

    node: FederationNode
    status: NodeStatus = NodeStatus.ONLINE
    incarnation: int = 0                      # Version counter
    last_update: float = field(default_factory=time.time)
    suspect_timeout: Optional[float] = None   # When to declare dead


# ═══════════════════════════════════════════════════════════════════════════
#  SWIM MEMBERSHIP
# ═══════════════════════════════════════════════════════════════════════════


class SwimMembership:
    """
    SWIM-style membership with suspicion protocol.

    Features:
    - Randomized probing (avoids hot spots)
    - Indirect probing (reduces false positives)
    - Suspicion mechanism (grace period before removal)
    - Piggybacked dissemination (efficient updates)

    Usage:
        membership = SwimMembership(my_node)
        await membership.join(["bootstrap.example.com:9000"])

        # Get random members for routing
        nodes = membership.get_random_members(k=5)

        # Check if a node is healthy
        healthy = await membership.ping(node_id)
    """

    def __init__(
        self,
        local_node: FederationNode,
        *,
        transport: Optional[BaseTransport] = None,
        ping_interval_ms: int = 1000,
        ping_timeout_ms: int = 500,
        suspicion_multiplier: int = 5,
        indirect_probes: int = 3,
        max_gossip_nodes: int = 5,
        on_join: Optional[Callable[[FederationNode], None]] = None,
        on_leave: Optional[Callable[[str], None]] = None,
    ):
        self._local = local_node
        self._incarnation = 0
        self._transport = transport

        # Timing
        self._ping_interval = ping_interval_ms / 1000
        self._ping_timeout = ping_timeout_ms / 1000
        self._suspicion_timeout = (ping_interval_ms * suspicion_multiplier) / 1000
        self._indirect_probes = indirect_probes
        self._max_gossip = max_gossip_nodes

        # State
        self._members: Dict[str, MemberState] = {}
        self._pending_pings: Dict[str, asyncio.Future] = {}
        self._update_queue: List[GossipMessage] = []

        # Callbacks
        self._on_join = on_join
        self._on_leave = on_leave

        # Background tasks
        self._protocol_task: Optional[asyncio.Task] = None
        self._running = False

    # ───────────────────────────────────────────────────────────────────────
    #  LIFECYCLE
    # ───────────────────────────────────────────────────────────────────────

    async def join(self, bootstrap_nodes: List[str]) -> bool:
        """
        Join the network via bootstrap nodes.

        Returns:
            True if successfully joined
        """
        logger.info(f"Joining via {len(bootstrap_nodes)} bootstrap nodes")

        # Start protocol loop
        self._running = True
        self._protocol_task = asyncio.create_task(self._protocol_loop())

        # TODO: Connect to bootstrap and sync membership
        # For now, just mark as joined

        return True

    async def leave(self) -> None:
        """Announce departure and leave gracefully."""
        logger.info("Leaving membership")

        # Broadcast leave message
        await self._broadcast(
            GossipMessage(
                type=MessageType.LEAVE,
                sender=self._local.node_id,
                target=self._local.node_id,
                incarnation=self._incarnation,
            )
        )

        # Stop protocol loop
        self._running = False
        if self._protocol_task:
            self._protocol_task.cancel()
            try:
                await self._protocol_task
            except asyncio.CancelledError:
                pass

    # ───────────────────────────────────────────────────────────────────────
    #  PROBING
    # ───────────────────────────────────────────────────────────────────────

    async def ping(self, node_id: str) -> bool:
        """
        Direct probe for liveness.

        Returns:
            True if node responded
        """
        if node_id not in self._members:
            return False

        member = self._members[node_id]

        try:
            # Create pending future
            future: asyncio.Future = asyncio.get_event_loop().create_future()
            self._pending_pings[node_id] = future

            # Send ping
            await self._send(
                member.node,
                GossipMessage(
                    type=MessageType.PING,
                    sender=self._local.node_id,
                    target=node_id,
                    incarnation=self._incarnation,
                ),
            )

            # Wait for ack
            await asyncio.wait_for(future, timeout=self._ping_timeout)
            return True

        except asyncio.TimeoutError:
            return False

        finally:
            self._pending_pings.pop(node_id, None)

    async def ping_req(
        self,
        target: str,
        intermediaries: List[str],
    ) -> bool:
        """
        Indirect probe via intermediaries.

        If direct ping fails, ask other nodes to probe on our behalf.
        Reduces false positives from network partitions.

        Returns:
            True if any intermediary got a response
        """
        if not intermediaries:
            return False

        # Ask each intermediary to ping the target
        tasks = []
        for node_id in intermediaries:
            if node_id in self._members:
                tasks.append(self._request_indirect_ping(node_id, target))

        if not tasks:
            return False

        # Return True if any succeeded
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return any(r is True for r in results)

    async def _request_indirect_ping(
        self,
        intermediary_id: str,
        target_id: str,
    ) -> bool:
        """Request an intermediary to ping the target."""
        if intermediary_id not in self._members:
            return False

        intermediary = self._members[intermediary_id]

        try:
            # Create pending future
            key = f"{intermediary_id}:{target_id}"
            future: asyncio.Future = asyncio.get_event_loop().create_future()
            self._pending_pings[key] = future

            # Send ping request
            await self._send(
                intermediary.node,
                GossipMessage(
                    type=MessageType.PING_REQ,
                    sender=self._local.node_id,
                    target=target_id,
                    incarnation=self._incarnation,
                ),
            )

            # Wait for ack
            await asyncio.wait_for(future, timeout=self._ping_timeout * 2)
            return True

        except asyncio.TimeoutError:
            return False

        finally:
            self._pending_pings.pop(f"{intermediary_id}:{target_id}", None)

    # ───────────────────────────────────────────────────────────────────────
    #  SUSPICION
    # ───────────────────────────────────────────────────────────────────────

    async def suspect(self, node_id: str) -> None:
        """Mark node as suspected (grace period before removal)."""
        if node_id not in self._members:
            return

        member = self._members[node_id]

        if member.status == NodeStatus.SUSPECT:
            return  # Already suspected

        logger.warning(f"Suspecting node {node_id[:8]}...")

        member.status = NodeStatus.SUSPECT
        member.suspect_timeout = time.time() + self._suspicion_timeout

        # Broadcast suspicion
        await self._broadcast(
            GossipMessage(
                type=MessageType.SUSPECT,
                sender=self._local.node_id,
                target=node_id,
                incarnation=member.incarnation,
            )
        )

    async def confirm_dead(self, node_id: str) -> None:
        """Confirm node failure after suspicion timeout."""
        if node_id not in self._members:
            return

        member = self._members[node_id]
        logger.info(f"Confirming node {node_id[:8]}... is dead")

        member.status = NodeStatus.OFFLINE

        # Broadcast death confirmation
        await self._broadcast(
            GossipMessage(
                type=MessageType.DEAD,
                sender=self._local.node_id,
                target=node_id,
                incarnation=member.incarnation,
            )
        )

        # Remove from membership
        del self._members[node_id]

        # Callback
        if self._on_leave:
            self._on_leave(node_id)

    async def refute_suspicion(self) -> None:
        """Refute suspicion about ourselves (if we're still alive)."""
        self._incarnation += 1

        await self._broadcast(
            GossipMessage(
                type=MessageType.ALIVE,
                sender=self._local.node_id,
                target=self._local.node_id,
                incarnation=self._incarnation,
            )
        )

    # ───────────────────────────────────────────────────────────────────────
    #  MEMBERSHIP QUERIES
    # ───────────────────────────────────────────────────────────────────────

    def get_members(self) -> List[FederationNode]:
        """Get all known members."""
        return [m.node for m in self._members.values() if m.status == NodeStatus.ONLINE]

    def get_random_members(self, k: int) -> List[FederationNode]:
        """Get k random healthy members."""
        healthy = self.get_members()
        return random.sample(healthy, min(k, len(healthy)))

    def get_member(self, node_id: str) -> Optional[FederationNode]:
        """Get a specific member."""
        if node_id in self._members:
            return self._members[node_id].node
        return None

    def is_healthy(self, node_id: str) -> bool:
        """Check if a node is healthy."""
        if node_id not in self._members:
            return False
        return self._members[node_id].status == NodeStatus.ONLINE

    @property
    def member_count(self) -> int:
        """Number of known members."""
        return len(self._members)

    # ───────────────────────────────────────────────────────────────────────
    #  PROTOCOL LOOP
    # ───────────────────────────────────────────────────────────────────────

    async def _protocol_loop(self) -> None:
        """Main SWIM protocol loop."""
        while self._running:
            try:
                # Pick a random member to probe
                members = self.get_members()
                if members:
                    target = random.choice(members)
                    await self._probe_member(target)

                # Check suspicion timeouts
                await self._check_suspicion_timeouts()

                # Sleep until next round
                await asyncio.sleep(self._ping_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Protocol loop error: {e}")
                await asyncio.sleep(1)

    async def _probe_member(self, node: FederationNode) -> None:
        """Probe a member and handle failure."""
        node_id = node.node_id

        # Try direct ping
        if await self.ping(node_id):
            return  # Healthy

        # Direct failed, try indirect
        intermediaries = [
            m.node_id
            for m in random.sample(
                list(self._members.values()),
                min(self._indirect_probes, len(self._members)),
            )
            if m.node_id != node_id
        ]

        if await self.ping_req(node_id, intermediaries):
            return  # Indirect succeeded

        # Both failed, suspect the node
        await self.suspect(node_id)

    async def _check_suspicion_timeouts(self) -> None:
        """Check and process suspicion timeouts."""
        now = time.time()

        for node_id, member in list(self._members.items()):
            if (
                member.status == NodeStatus.SUSPECT
                and member.suspect_timeout
                and now > member.suspect_timeout
            ):
                await self.confirm_dead(node_id)

    # ───────────────────────────────────────────────────────────────────────
    #  MESSAGE HANDLING
    # ───────────────────────────────────────────────────────────────────────

    async def handle_message(self, message: GossipMessage) -> None:
        """Handle incoming gossip message."""
        handlers = {
            MessageType.PING: self._handle_ping,
            MessageType.PING_REQ: self._handle_ping_req,
            MessageType.ACK: self._handle_ack,
            MessageType.ALIVE: self._handle_alive,
            MessageType.SUSPECT: self._handle_suspect,
            MessageType.DEAD: self._handle_dead,
            MessageType.JOIN: self._handle_join,
            MessageType.LEAVE: self._handle_leave,
        }

        handler = handlers.get(message.type)
        if handler:
            await handler(message)

    async def _handle_ping(self, msg: GossipMessage) -> None:
        """Handle direct ping - respond with ack."""
        if msg.sender in self._members:
            sender = self._members[msg.sender].node
            await self._send(
                sender,
                GossipMessage(
                    type=MessageType.ACK,
                    sender=self._local.node_id,
                    target=msg.sender,
                    incarnation=self._incarnation,
                ),
            )

    async def _handle_ping_req(self, msg: GossipMessage) -> None:
        """Handle indirect ping request."""
        # Ping the target
        success = await self.ping(msg.target)

        # Reply to requester
        if msg.sender in self._members:
            sender = self._members[msg.sender].node
            await self._send(
                sender,
                GossipMessage(
                    type=MessageType.ACK,
                    sender=self._local.node_id,
                    target=msg.sender,
                    incarnation=self._incarnation,
                    payload={"target": msg.target, "success": success},
                ),
            )

    async def _handle_ack(self, msg: GossipMessage) -> None:
        """Handle probe acknowledgment."""
        # Check for direct ping
        if msg.sender in self._pending_pings:
            future = self._pending_pings[msg.sender]
            if not future.done():
                future.set_result(True)

        # Check for indirect ping
        target = msg.payload.get("target")
        if target:
            key = f"{msg.sender}:{target}"
            if key in self._pending_pings:
                future = self._pending_pings[key]
                if not future.done() and msg.payload.get("success"):
                    future.set_result(True)

    async def _handle_alive(self, msg: GossipMessage) -> None:
        """Handle alive message (refutes suspicion)."""
        if msg.target in self._members:
            member = self._members[msg.target]
            if msg.incarnation > member.incarnation:
                member.incarnation = msg.incarnation
                member.status = NodeStatus.ONLINE
                member.suspect_timeout = None
                logger.info(f"Node {msg.target[:8]}... is alive")

    async def _handle_suspect(self, msg: GossipMessage) -> None:
        """Handle suspicion message."""
        if msg.target == self._local.node_id:
            # We're being suspected - refute!
            await self.refute_suspicion()
        elif msg.target in self._members:
            member = self._members[msg.target]
            if msg.incarnation >= member.incarnation:
                await self.suspect(msg.target)

    async def _handle_dead(self, msg: GossipMessage) -> None:
        """Handle death confirmation."""
        if msg.target in self._members:
            await self.confirm_dead(msg.target)

    async def _handle_join(self, msg: GossipMessage) -> None:
        """Handle join announcement."""
        # TODO: Add node to membership
        pass

    async def _handle_leave(self, msg: GossipMessage) -> None:
        """Handle leave announcement."""
        if msg.target in self._members:
            del self._members[msg.target]
            if self._on_leave:
                self._on_leave(msg.target)

    # ───────────────────────────────────────────────────────────────────────
    #  TRANSPORT
    # ───────────────────────────────────────────────────────────────────────

    def _gossip_to_transport_type(self, msg_type: MessageType) -> TransportMessageType:
        """Map gossip MessageType to transport MessageType."""
        mapping = {
            MessageType.PING: TransportMessageType.PING,
            MessageType.PING_REQ: TransportMessageType.PING_REQ,
            MessageType.ACK: TransportMessageType.ACK,
            MessageType.ALIVE: TransportMessageType.ALIVE,
            MessageType.SUSPECT: TransportMessageType.SUSPECT,
            MessageType.DEAD: TransportMessageType.DEAD,
            MessageType.JOIN: TransportMessageType.JOIN,
            MessageType.LEAVE: TransportMessageType.LEAVE,
        }
        return mapping[msg_type]

    def _gossip_to_transport_message(self, message: GossipMessage) -> TransportMessage:
        """Convert GossipMessage to TransportMessage for wire transport."""
        return TransportMessage(
            type=self._gossip_to_transport_type(message.type),
            sender=message.sender,
            payload={
                "target": message.target,
                "incarnation": message.incarnation,
                **message.payload,
            },
        )

    async def _send(self, node: FederationNode, message: GossipMessage) -> bool:
        """
        Send message to a node via transport layer.

        Args:
            node: Target node
            message: Gossip message to send

        Returns:
            True if send succeeded
        """
        if not self._transport:
            logger.warning("No transport configured, message not sent")
            return False

        transport_msg = self._gossip_to_transport_message(message)

        try:
            response = await self._transport.send(
                node,
                transport_msg,
                timeout_ms=int(self._ping_timeout * 1000),
            )
            return response.success
        except Exception as e:
            logger.debug(f"Send to {node.node_id[:8]}... failed: {e}")
            return False

    async def _broadcast(self, message: GossipMessage) -> int:
        """
        Broadcast message to random subset of members (parallel).

        Returns:
            Number of successful sends
        """
        targets = self.get_random_members(self._max_gossip)
        if not targets:
            return 0

        # Send to all targets in parallel
        tasks = [self._send(node, message) for node in targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes
        return sum(1 for r in results if r is True)
