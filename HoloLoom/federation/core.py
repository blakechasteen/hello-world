"""
Federation Core - The one class you need.

async with Federation() as fed:
    await fed.join("bootstrap.hololoom.net:9000")
    result = await fed.query("What is love?")

That's it. That's the API.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from HoloLoom.hololoom import HoloLoom as HoloLoomType

from .gossip import SwimMembership
from .identity import Identity, create_node, get_or_create_identity
from .protocols import FederationProtocol
from .routing import KademliaRouter
from .transport import (
    BaseTransport,
    create_http_transport,
    MessageType,
    TransportMessage,
)
from .types import (
    Capability,
    FederationError,
    FederationNode,
    Guild,
    NetworkError,
    Query,
    QueryTrace,
    Response,
    RoutingError,
    TimeoutError,
    Verification,
    VerificationLevel,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION - Sane defaults, full control
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class FederationConfig:
    """
    Federation configuration.

    Most users never need to change these.
    """

    # Identity
    identity_path: Optional[str] = None       # Path to key file (auto-generates if None)
    endpoint: str = "0.0.0.0:9000"           # Listen address

    # Network
    default_timeout_ms: int = 5000            # Query timeout
    connection_timeout_ms: int = 3000         # Connection timeout
    max_connections: int = 100                # Connection pool size

    # Verification
    default_level: VerificationLevel = VerificationLevel.STANDARD
    verification_timeout_ms: int = 10000      # Max time for verification

    # Gossip
    gossip_interval_ms: int = 1000            # Heartbeat interval
    suspicion_timeout_ms: int = 5000          # Grace period before declaring dead
    max_gossip_peers: int = 5                 # Peers per gossip round

    # Routing
    k_bucket_size: int = 20                   # Kademlia k parameter
    alpha: int = 3                            # Kademlia α (parallel lookups)
    min_trust_score: float = 0.5              # Minimum trust for routing

    # Capabilities
    capabilities: Set[Capability] = field(
        default_factory=lambda: {Capability.WEAVING}
    )

    # Local fallback
    fallback_to_local: bool = True            # Use local HoloLoom if network unavailable

    @classmethod
    def default(cls) -> "FederationConfig":
        """Default configuration for most users."""
        return cls()

    @classmethod
    def development(cls) -> "FederationConfig":
        """Relaxed settings for development."""
        return cls(
            default_timeout_ms=30000,
            verification_timeout_ms=60000,
            min_trust_score=0.0,  # Accept any node
        )

    @classmethod
    def production(cls) -> "FederationConfig":
        """Strict settings for production."""
        return cls(
            default_timeout_ms=3000,
            connection_timeout_ms=2000,
            min_trust_score=0.7,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  FEDERATION - The main class
# ═══════════════════════════════════════════════════════════════════════════


class Federation:
    """
    The federation client.

    This is the only class most users need. It:
    - Manages node identity (Ed25519)
    - Handles network connectivity (gRPC)
    - Routes queries to capable nodes (Kademlia DHT)
    - Verifies responses through consensus (SWIM gossip)
    - Manages guild memberships (trust groups)

    Usage:
        # Simple
        async with Federation() as fed:
            await fed.join("bootstrap.hololoom.net:9000")
            result = await fed.query("What is quantum computing?")
            print(result.answer)

        # With options
        async with Federation(config=FederationConfig.production()) as fed:
            await fed.join("bootstrap.hololoom.net:9000")
            await fed.join_guild("medical")

            result = await fed.query(
                "Is this medication safe?",
                level=VerificationLevel.CRITICAL,
                guild="medical"
            )
    """

    def __init__(
        self,
        config: Optional[FederationConfig] = None,
        identity: Optional[Identity] = None,
        loom: Optional["HoloLoomType"] = None,
    ):
        self._config = config or FederationConfig.default()
        self._identity = identity
        self._node: Optional[FederationNode] = None
        self._loom: Optional["HoloLoomType"] = loom

        # Components (initialized on connect)
        self._membership: Optional[Any] = None  # MembershipProtocol
        self._router: Optional[Any] = None      # RoutingProtocol
        self._verifier: Optional[Any] = None    # VerificationProtocol
        self._guilds: Optional[Any] = None      # GuildProtocol
        self._transport: Optional[Any] = None   # TransportProtocol

        # State
        self._connected = False
        self._peers: Dict[str, FederationNode] = {}
        self._my_guilds: Set[str] = set()
        self._metrics: Dict[str, Any] = {}
        self._started_at: Optional[datetime] = None

    # ───────────────────────────────────────────────────────────────────────
    #  LIFECYCLE
    # ───────────────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "Federation":
        """Async context manager entry."""
        await self._initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.leave()

    async def _initialize(self) -> None:
        """Initialize identity and local components."""
        # Get or create identity
        if self._identity is None:
            if self._config.identity_path:
                self._identity = Identity.load(self._config.identity_path)
            else:
                self._identity = get_or_create_identity()

        # Create our node representation
        self._node = create_node(
            self._identity,
            self._config.endpoint,
            capabilities=self._config.capabilities,
        )

        self._started_at = datetime.utcnow()
        logger.info(f"Federation initialized: {self._identity.short_id}")

    # ───────────────────────────────────────────────────────────────────────
    #  NETWORK
    # ───────────────────────────────────────────────────────────────────────

    async def join(self, bootstrap: str) -> None:
        """
        Join the federation network.

        Args:
            bootstrap: Bootstrap node address (host:port)

        Raises:
            NetworkError: If unable to connect
        """
        if self._connected:
            logger.warning("Already connected to federation")
            return

        logger.info(f"Joining federation via {bootstrap}...")

        try:
            # 1. Create and start transport
            self._transport = create_http_transport(
                host=self._config.endpoint.split(":")[0],
                port=int(self._config.endpoint.split(":")[1]),
                timeout_ms=self._config.connection_timeout_ms,
                max_connections=self._config.max_connections,
            )
            await self._transport.start()

            # 2. Initialize SWIM membership with transport
            self._membership = SwimMembership(
                local_node=self._node,
                ping_interval_ms=self._config.gossip_interval_ms,
                ping_timeout_ms=self._config.suspicion_timeout_ms,
                max_gossip_peers=self._config.max_gossip_peers,
                transport=self._transport,
            )

            # 3. Initialize Kademlia router with transport
            self._router = KademliaRouter(
                local_node=self._node,
                k=self._config.k_bucket_size,
                alpha=self._config.alpha,
                transport=self._transport,
            )

            # 4. Join via bootstrap node
            await self._membership.join(bootstrap)

            # 5. Populate routing table with discovered peers
            for peer in self._membership.get_members():
                self._router.update(peer)
                self._peers[peer.node_id] = peer

            self._connected = True
            logger.info(f"Joined federation as {self.node_id[:8]}... ({len(self._peers)} peers)")

        except Exception as e:
            if self._config.fallback_to_local:
                logger.warning(f"Federation unavailable ({e}), using local mode")
                self._connected = False
            else:
                raise NetworkError(
                    f"Failed to join federation: {e}",
                    suggestion="Check network connectivity and bootstrap address",
                )

    async def leave(self) -> None:
        """Gracefully leave the network."""
        if not self._connected:
            return

        logger.info("Leaving federation...")

        # 1. Announce departure via membership (graceful leave)
        if self._membership:
            try:
                await self._membership.leave()
            except Exception as e:
                logger.warning(f"Error during membership leave: {e}")

        # 2. Stop transport server
        if self._transport:
            try:
                await self._transport.stop()
            except Exception as e:
                logger.warning(f"Error stopping transport: {e}")

        # 3. Clear state
        self._connected = False
        self._peers.clear()
        self._membership = None
        self._router = None
        self._transport = None

        logger.info("Left federation")

    # ───────────────────────────────────────────────────────────────────────
    #  QUERY
    # ───────────────────────────────────────────────────────────────────────

    async def query(
        self,
        text: str,
        *,
        verify: bool = True,
        level: Optional[VerificationLevel] = None,
        guild: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> "FederatedResponse":
        """
        Query the federation.

        Args:
            text: Query text
            verify: Whether to verify response (default: True)
            level: Verification level (default: STANDARD)
            guild: Preferred guild for routing
            timeout_ms: Query timeout in milliseconds

        Returns:
            FederatedResponse with answer, confidence, and provenance

        Raises:
            RoutingError: If no capable nodes found
            TimeoutError: If query times out
            VerificationError: If verification fails
        """
        level = level or self._config.default_level
        timeout_ms = timeout_ms or self._config.default_timeout_ms

        # Create query object
        query = Query(
            text=text,
            request_id=str(uuid.uuid4()),
            requester=self.node_id,
            level=level if verify else VerificationLevel.NONE,
            guild=guild,
            timeout_ms=timeout_ms,
        )

        # Start trace
        trace = QueryTrace(request_id=query.request_id)

        try:
            if self._connected:
                return await self._federated_query(query, trace)
            elif self._config.fallback_to_local:
                return await self._local_query(query, trace)
            else:
                raise NetworkError(
                    "Not connected to federation",
                    suggestion="Call await federation.join(bootstrap) first",
                )

        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Query timed out after {timeout_ms}ms",
                suggestion="Try increasing timeout or using a simpler query",
            )

    async def _federated_query(
        self,
        query: Query,
        trace: QueryTrace,
    ) -> "FederatedResponse":
        """Execute query through federation."""
        import time

        start = time.perf_counter()

        # TODO: Implement actual routing and verification
        # For now, return a placeholder

        # Simulate routing
        trace.add("route", self.node_id, 5.0, nodes_found=3)

        # Simulate verification
        trace.add("verify", self.node_id, 50.0, verifiers=3, agreed=True)

        # Simulate response
        trace.add("respond", self.node_id, 100.0)
        trace.complete()

        elapsed_ms = (time.perf_counter() - start) * 1000

        return FederatedResponse(
            answer="[Placeholder: Federation query processing not yet implemented]",
            confidence=0.0,
            verified=False,
            verified_by=[],
            trace=trace,
            source="federation",
            latency_ms=elapsed_ms,
        )

    async def _local_query(
        self,
        query: Query,
        trace: QueryTrace,
    ) -> "FederatedResponse":
        """Execute query locally using HoloLoom."""
        import time

        start = time.perf_counter()

        # If no local loom, return placeholder
        if self._loom is None:
            trace.add("local", self.node_id, 0.5, error="no_loom_instance")
            trace.complete()
            elapsed_ms = (time.perf_counter() - start) * 1000
            return FederatedResponse(
                answer="[Local mode: No HoloLoom instance configured]",
                confidence=0.0,
                verified=False,
                verified_by=[],
                trace=trace,
                source="local",
                latency_ms=elapsed_ms,
            )

        try:
            # Use HoloLoom's recall() for memory retrieval
            memories = await self._loom.recall(query.text, limit=10)

            recall_ms = (time.perf_counter() - start) * 1000
            trace.add("recall", self.node_id, recall_ms, memories_found=len(memories))

            # Build answer from retrieved memories
            if memories:
                # Combine top memories into response
                answer_parts = []
                total_confidence = 0.0
                for mem in memories[:5]:  # Top 5
                    answer_parts.append(mem.text if hasattr(mem, 'text') else str(mem))
                    # Get confidence from metadata or default
                    mem_conf = 0.5
                    if hasattr(mem, 'metadata') and isinstance(mem.metadata, dict):
                        mem_conf = mem.metadata.get('confidence', 0.5)
                    elif hasattr(mem, 'relevance'):
                        mem_conf = mem.relevance
                    total_confidence += mem_conf

                answer = "\n\n".join(answer_parts)
                confidence = total_confidence / min(len(memories), 5)
            else:
                answer = f"[No relevant memories found for: {query.text}]"
                confidence = 0.0

            trace.add("synthesize", self.node_id, 1.0)
            trace.complete()

            elapsed_ms = (time.perf_counter() - start) * 1000

            return FederatedResponse(
                answer=answer,
                confidence=confidence,
                verified=False,  # Local queries are not verified
                verified_by=[],
                trace=trace,
                source="local",
                latency_ms=elapsed_ms,
            )

        except Exception as e:
            logger.warning(f"Local query failed: {e}")
            trace.add("local", self.node_id, (time.perf_counter() - start) * 1000, error=str(e))
            trace.complete()

            return FederatedResponse(
                answer=f"[Local query error: {e}]",
                confidence=0.0,
                verified=False,
                verified_by=[],
                trace=trace,
                source="local",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

    async def verify(
        self,
        text: str,
        *,
        level: Optional[VerificationLevel] = None,
    ) -> Verification:
        """
        Request verification only (no response generation).

        Useful for verifying external claims.
        """
        level = level or self._config.default_level

        query = Query(
            text=text,
            request_id=str(uuid.uuid4()),
            requester=self.node_id,
            level=level,
        )

        # TODO: Implement verification-only flow
        return Verification(
            request_id=query.request_id,
            verified=False,
            confidence=0.0,
            consensus_response="",
            verifiers=frozenset(),
        )

    # ───────────────────────────────────────────────────────────────────────
    #  GUILDS
    # ───────────────────────────────────────────────────────────────────────

    async def join_guild(self, guild_id: str) -> bool:
        """
        Join a guild.

        Returns:
            True if joined successfully
        """
        if guild_id in self._my_guilds:
            return True

        # TODO: Implement guild joining
        self._my_guilds.add(guild_id)
        logger.info(f"Joined guild: {guild_id}")
        return True

    async def leave_guild(self, guild_id: str) -> None:
        """Leave a guild."""
        self._my_guilds.discard(guild_id)
        logger.info(f"Left guild: {guild_id}")

    async def list_guilds(self) -> List[Guild]:
        """List available guilds."""
        # TODO: Implement guild listing
        return []

    # ───────────────────────────────────────────────────────────────────────
    #  PROPERTIES
    # ───────────────────────────────────────────────────────────────────────

    @property
    def node_id(self) -> str:
        """This node's unique identifier."""
        if self._identity is None:
            raise RuntimeError("Federation not initialized")
        return self._identity.node_id

    @property
    def is_connected(self) -> bool:
        """Whether connected to the network."""
        return self._connected

    @property
    def peer_count(self) -> int:
        """Number of known peers."""
        return len(self._peers)

    @property
    def guilds(self) -> Set[str]:
        """Guilds this node belongs to."""
        return self._my_guilds.copy()

    def stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            "node_id": self.node_id[:8] + "...",
            "connected": self._connected,
            "peers": self.peer_count,
            "guilds": list(self._my_guilds),
            "uptime_seconds": (
                (datetime.utcnow() - self._started_at).total_seconds()
                if self._started_at
                else 0
            ),
            **self._metrics,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  RESPONSE - What you get back
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class FederatedResponse:
    """
    Response from a federated query.

    Contains the answer plus complete provenance.
    """

    answer: str
    confidence: float                         # 0.0-1.0
    verified: bool                            # Whether consensus was reached
    verified_by: List[str]                    # Node IDs that verified
    trace: QueryTrace                         # Full execution trace
    source: str                               # "federation" or "local"
    latency_ms: float                         # Total latency

    def __str__(self) -> str:
        status = "✓" if self.verified else "○"
        return f"{status} ({self.confidence:.0%}) {self.answer[:100]}..."

    def __repr__(self) -> str:
        return (
            f"FederatedResponse("
            f"verified={self.verified}, "
            f"confidence={self.confidence:.2f}, "
            f"verifiers={len(self.verified_by)}, "
            f"source={self.source!r})"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE - Quick start
# ═══════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def connect(
    bootstrap: str,
    config: Optional[FederationConfig] = None,
    loom: Optional["HoloLoomType"] = None,
) -> AsyncIterator[Federation]:
    """
    Quick connect to federation with optional local HoloLoom.

    Usage:
        async with connect("bootstrap.hololoom.net:9000") as fed:
            result = await fed.query("Hello world")

        # With local fallback
        async with connect("bootstrap.hololoom.net:9000", loom=my_loom) as fed:
            result = await fed.query("Hello world")  # Uses loom if network fails
    """
    async with Federation(config=config, loom=loom) as fed:
        await fed.join(bootstrap)
        yield fed
