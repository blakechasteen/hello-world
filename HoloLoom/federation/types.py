"""
Federation Types - The vocabulary of the network.

Simple, immutable data structures. No logic here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set


# ═══════════════════════════════════════════════════════════════════════════
#  ENUMS - The language of the network
# ═══════════════════════════════════════════════════════════════════════════


class NodeStatus(Enum):
    """Node health status."""

    ONLINE = auto()      # Healthy and responding
    DEGRADED = auto()    # Slow but functional
    SUSPECT = auto()     # Suspected failure (grace period)
    OFFLINE = auto()     # Confirmed unreachable


class VerificationLevel(Enum):
    """How thoroughly to verify responses."""

    NONE = 0         # Skip verification (internal/trusted)
    LIGHT = 2        # Quick check (2 verifiers)
    STANDARD = 3     # Default (3 verifiers)
    DEEP = 5         # Thorough (5 verifiers)
    CRITICAL = 7     # Maximum + human review (7+ verifiers)


class Capability(Enum):
    """What a node can do."""

    WEAVING = auto()     # HoloLoom weaving
    RAG = auto()         # Retrieval-augmented generation
    AGENTIC = auto()     # Multi-step reasoning
    CODE = auto()        # Code analysis/generation
    MEDICAL = auto()     # Medical domain expertise
    LEGAL = auto()       # Legal domain expertise
    RESEARCH = auto()    # Deep research mode
    EMBEDDING = auto()   # Embedding generation


class GuildTrustLevel(Enum):
    """Guild maturity and trust."""

    STARTER = auto()       # <30 days, quorum=5
    ESTABLISHED = auto()   # 30-180 days, quorum=3
    VETERAN = auto()       # >180 days, quorum=2


class AdmissionPolicy(Enum):
    """How new members join a guild."""

    OPEN = auto()         # Anyone can join
    VOUCHED = auto()      # Needs sponsor from existing member
    VOTED = auto()        # Requires majority vote
    CLOSED = auto()       # No new members


# ═══════════════════════════════════════════════════════════════════════════
#  CORE DATA - What flows through the network
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class Query:
    """A question to the federation."""

    text: str
    request_id: str
    requester: str                           # Node ID
    level: VerificationLevel = VerificationLevel.STANDARD
    guild: Optional[str] = None              # Preferred guild
    timeout_ms: int = 5000
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True, slots=True)
class Response:
    """An answer from a node."""

    text: str
    request_id: str
    responder: str                           # Node ID
    confidence: float                        # 0.0-1.0
    latency_ms: float
    signature: bytes = b""                   # Ed25519 signature
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True, slots=True)
class Verification:
    """Consensus verification result."""

    request_id: str
    verified: bool
    confidence: float                        # Agreement level
    consensus_response: str                  # Merged/selected answer
    verifiers: FrozenSet[str]               # Node IDs that verified
    dissenting: FrozenSet[str] = frozenset() # Nodes that disagreed
    scores: Dict[str, float] = field(default_factory=dict)  # DS-STAR scores
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def quorum_met(self) -> bool:
        """Did we meet quorum?"""
        return len(self.verifiers) >= 2  # Minimum for consensus


# ═══════════════════════════════════════════════════════════════════════════
#  NODE & GUILD - The participants
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class FederationNode:
    """A node in the federation."""

    node_id: str                             # SHA256(public_key)[:20]
    public_key: bytes                        # Ed25519 public key
    endpoint: str                            # host:port
    capabilities: Set[Capability] = field(default_factory=set)
    guilds: Set[str] = field(default_factory=set)
    trust_score: float = 0.5                 # 0.0-1.0, starts neutral
    version: str = "1.0.0"
    status: NodeStatus = NodeStatus.ONLINE
    last_seen: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FederationNode):
            return False
        return self.node_id == other.node_id


@dataclass(slots=True)
class Guild:
    """A trust group for domain specialization."""

    guild_id: str
    name: str
    domain: str
    trust_level: GuildTrustLevel = GuildTrustLevel.STARTER
    admission: AdmissionPolicy = AdmissionPolicy.OPEN
    members: Set[str] = field(default_factory=set)        # Node IDs
    reputation: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def quorum(self) -> int:
        """Required verifiers based on trust level."""
        return {
            GuildTrustLevel.STARTER: 5,
            GuildTrustLevel.ESTABLISHED: 3,
            GuildTrustLevel.VETERAN: 2,
        }[self.trust_level]

    def __hash__(self) -> int:
        return hash(self.guild_id)


# ═══════════════════════════════════════════════════════════════════════════
#  NETWORK EVENTS - What happens
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class NodeJoined:
    """A node joined the network."""

    node: FederationNode
    bootstrap: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True, slots=True)
class NodeLeft:
    """A node left the network."""

    node_id: str
    graceful: bool                           # Clean departure vs timeout
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True, slots=True)
class NodeSuspected:
    """A node is suspected of failure."""

    node_id: str
    reporter: str                            # Who reported it
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True, slots=True)
class VerificationComplete:
    """Verification finished."""

    verification: Verification
    query: Query
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════════════════
#  TRACE - Full provenance
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class TraceStep:
    """One step in query processing."""

    action: str                              # "route", "verify", "respond", etc.
    node_id: str
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class QueryTrace:
    """Complete trace of a query's journey."""

    request_id: str
    steps: List[TraceStep] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    def add(
        self,
        action: str,
        node_id: str,
        duration_ms: float,
        **details: Any,
    ) -> None:
        """Add a trace step."""
        self.steps.append(
            TraceStep(
                action=action,
                node_id=node_id,
                duration_ms=duration_ms,
                details=details,
            )
        )

    def complete(self) -> None:
        """Mark trace as complete."""
        self.completed_at = datetime.utcnow()

    @property
    def total_ms(self) -> float:
        """Total duration in milliseconds."""
        return sum(s.duration_ms for s in self.steps)

    def __str__(self) -> str:
        """Pretty print the trace."""
        lines = [f"Query {self.request_id}:"]
        for step in self.steps:
            lines.append(f"  {step.action} @ {step.node_id} ({step.duration_ms:.1f}ms)")
        lines.append(f"  Total: {self.total_ms:.1f}ms")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  ERRORS - When things go wrong
# ═══════════════════════════════════════════════════════════════════════════


class FederationError(Exception):
    """Base federation error with helpful messages."""

    def __init__(
        self,
        message: str,
        *,
        node_id: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        self.node_id = node_id
        self.suggestion = suggestion
        super().__init__(self._format_message(message))

    def _format_message(self, message: str) -> str:
        parts = [message]
        if self.node_id:
            parts.append(f"Node: {self.node_id}")
        if self.suggestion:
            parts.append(f"Try: {self.suggestion}")
        return " | ".join(parts)


class NetworkError(FederationError):
    """Network-related failure."""

    pass


class VerificationError(FederationError):
    """Verification failed to reach consensus."""

    pass


class RoutingError(FederationError):
    """Could not route to capable nodes."""

    pass


class GuildError(FederationError):
    """Guild operation failed."""

    pass


class TimeoutError(FederationError):
    """Operation timed out."""

    pass
