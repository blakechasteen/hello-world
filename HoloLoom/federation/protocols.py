"""
Federation Protocols - The contracts that bind the network.

"Perfection is achieved not when there is nothing more to add,
 but when there is nothing left to take away." — Antoine de Saint-Exupéry

All protocols in one place. Implementations come later.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    List,
    Optional,
    Protocol,
    Set,
    runtime_checkable,
)

if TYPE_CHECKING:
    from .types import (
        Capability,
        FederationNode,
        Guild,
        NodeStatus,
        Query,
        Response,
        Verification,
        VerificationLevel,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  CORE PROTOCOL - The one interface that matters
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class FederationProtocol(Protocol):
    """
    The complete federation interface.

    This is the only protocol users need to know.
    Everything else is implementation detail.

    Usage:
        async with Federation() as fed:
            await fed.join("bootstrap.hololoom.net:9000")
            result = await fed.query("What is love?")
    """

    @abstractmethod
    async def join(self, bootstrap: str) -> None:
        """Join the network via a bootstrap node."""
        ...

    @abstractmethod
    async def leave(self) -> None:
        """Gracefully leave the network."""
        ...

    @abstractmethod
    async def query(
        self,
        text: str,
        *,
        verify: bool = True,
        level: Optional[VerificationLevel] = None,
        guild: Optional[str] = None,
        timeout_ms: int = 5000,
    ) -> Response:
        """
        Query the federation.

        Args:
            text: The query text
            verify: Whether to verify response (default: True)
            level: Verification level (default: STANDARD)
            guild: Preferred guild for routing
            timeout_ms: Query timeout in milliseconds

        Returns:
            Response with answer, confidence, and provenance
        """
        ...

    @abstractmethod
    async def verify(
        self,
        text: str,
        *,
        level: Optional[VerificationLevel] = None,
    ) -> Verification:
        """Request verification only (no response generation)."""
        ...

    @property
    @abstractmethod
    def node_id(self) -> str:
        """This node's unique identifier."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether connected to the network."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
#  TRANSPORT LAYER - How nodes talk
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class TransportProtocol(Protocol):
    """Wire protocol for node-to-node communication."""

    @abstractmethod
    async def connect(self, endpoint: str) -> None:
        """Establish connection to endpoint."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection gracefully."""
        ...

    @abstractmethod
    async def send(self, message: bytes) -> None:
        """Send raw message."""
        ...

    @abstractmethod
    async def recv(self) -> bytes:
        """Receive raw message."""
        ...

    @abstractmethod
    async def stream(self, message: bytes) -> AsyncIterator[bytes]:
        """Bidirectional streaming."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Connection status."""
        ...


@runtime_checkable
class ConnectionPoolProtocol(Protocol):
    """Manages connections to multiple peers."""

    @abstractmethod
    async def get(self, endpoint: str) -> TransportProtocol:
        """Get or create connection to endpoint."""
        ...

    @abstractmethod
    async def release(self, endpoint: str) -> None:
        """Release connection back to pool."""
        ...

    @abstractmethod
    async def close_all(self) -> None:
        """Close all connections."""
        ...

    @property
    @abstractmethod
    def active_connections(self) -> int:
        """Number of active connections."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
#  MEMBERSHIP LAYER - Who's in the network (SWIM gossip)
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class MembershipProtocol(Protocol):
    """SWIM-style membership with failure detection."""

    @abstractmethod
    async def join(self, bootstrap_nodes: List[str]) -> bool:
        """Join network via bootstrap nodes."""
        ...

    @abstractmethod
    async def leave(self) -> None:
        """Announce departure and leave gracefully."""
        ...

    @abstractmethod
    async def ping(self, node_id: str) -> bool:
        """Direct probe for liveness."""
        ...

    @abstractmethod
    async def ping_req(
        self,
        target: str,
        intermediaries: List[str],
    ) -> bool:
        """Indirect probe via intermediaries."""
        ...

    @abstractmethod
    async def suspect(self, node_id: str) -> None:
        """Mark node as suspected (grace period before removal)."""
        ...

    @abstractmethod
    async def confirm_dead(self, node_id: str) -> None:
        """Confirm node failure after suspicion timeout."""
        ...

    @abstractmethod
    def get_members(self) -> List[FederationNode]:
        """Get current membership list."""
        ...

    @abstractmethod
    def get_random_members(self, k: int) -> List[FederationNode]:
        """Get k random members for gossip."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTING LAYER - Finding the right nodes (Kademlia DHT)
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class RoutingProtocol(Protocol):
    """Kademlia-style DHT for capability-based routing."""

    @abstractmethod
    async def find_nodes(
        self,
        capabilities: Set[Capability],
        *,
        min_trust: float = 0.7,
        guild: Optional[str] = None,
        limit: int = 10,
    ) -> List[FederationNode]:
        """
        Find nodes with required capabilities.

        Args:
            capabilities: Required capabilities
            min_trust: Minimum trust score (0.0-1.0)
            guild: Preferred guild (optional)
            limit: Maximum nodes to return

        Returns:
            List of capable nodes, sorted by trust score
        """
        ...

    @abstractmethod
    async def route(
        self,
        query: Query,
        nodes: List[FederationNode],
    ) -> List[Response]:
        """Route query to nodes and collect responses."""
        ...

    @abstractmethod
    async def announce(
        self,
        node: FederationNode,
    ) -> None:
        """Announce node capabilities to DHT."""
        ...

    @abstractmethod
    async def lookup(self, node_id: str) -> Optional[FederationNode]:
        """Lookup node by ID."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
#  VERIFICATION LAYER - Safety through consensus
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class VerificationProtocol(Protocol):
    """Multi-node consensus verification."""

    @abstractmethod
    async def verify(
        self,
        query: Query,
        responses: List[Response],
        *,
        level: VerificationLevel,
        quorum: int,
    ) -> Verification:
        """
        Verify responses through consensus.

        Args:
            query: Original query
            responses: Candidate responses from nodes
            level: Verification thoroughness
            quorum: Minimum agreeing nodes

        Returns:
            Verification result with consensus response
        """
        ...

    @abstractmethod
    async def vote(
        self,
        query: Query,
        response: Response,
    ) -> float:
        """
        Vote on a response's quality.

        Returns:
            Vote score (0.0-1.0)
        """
        ...

    @abstractmethod
    async def check_agreement(
        self,
        responses: List[Response],
        threshold: float = 0.66,
    ) -> bool:
        """Check if responses agree above threshold."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
#  GUILD LAYER - Trust through specialization
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class GuildProtocol(Protocol):
    """Guild management and trust."""

    @abstractmethod
    async def create(
        self,
        name: str,
        domain: str,
        *,
        admission_policy: str = "open",
    ) -> Guild:
        """Create a new guild."""
        ...

    @abstractmethod
    async def join(self, guild_id: str) -> bool:
        """Request to join a guild."""
        ...

    @abstractmethod
    async def leave(self, guild_id: str) -> None:
        """Leave a guild."""
        ...

    @abstractmethod
    async def get_members(self, guild_id: str) -> List[FederationNode]:
        """Get guild members."""
        ...

    @abstractmethod
    async def update_reputation(
        self,
        guild_id: str,
        node_id: str,
        delta: float,
    ) -> float:
        """
        Update node reputation in guild.

        Returns:
            New reputation score
        """
        ...

    @abstractmethod
    async def get_quorum(self, guild_id: str) -> int:
        """Get required quorum for guild's trust level."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
#  IDENTITY - Cryptographic identity
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class IdentityProtocol(Protocol):
    """Ed25519 cryptographic identity."""

    @abstractmethod
    def sign(self, message: bytes) -> bytes:
        """Sign message with private key."""
        ...

    @abstractmethod
    def verify_signature(
        self,
        message: bytes,
        signature: bytes,
        public_key: bytes,
    ) -> bool:
        """Verify signature from public key."""
        ...

    @property
    @abstractmethod
    def node_id(self) -> str:
        """Derived node ID (SHA256(public_key)[:20])."""
        ...

    @property
    @abstractmethod
    def public_key(self) -> bytes:
        """Ed25519 public key."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
#  OBSERVABILITY - See what's happening
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class MetricsProtocol(Protocol):
    """Observable metrics for monitoring."""

    @abstractmethod
    def record_query(
        self,
        query: Query,
        response: Response,
        latency_ms: float,
    ) -> None:
        """Record query metrics."""
        ...

    @abstractmethod
    def record_verification(
        self,
        verification: Verification,
        latency_ms: float,
    ) -> None:
        """Record verification metrics."""
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get current statistics."""
        ...

    @abstractmethod
    def export_prometheus(self) -> str:
        """Export Prometheus-format metrics."""
        ...
