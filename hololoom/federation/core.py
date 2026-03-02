"""
Federation Core - The one class you need.

async with Federation() as fed:
    await fed.join("bootstrap.hololoom.net:9000")
    result = await fed.query("What is love?")

That's it. That's the API.

Every federated response carries alignment proof.
Every receiving node verifies alignment before accepting.
Philosophy: "Don't trust, verify alignment" - Byzantine safety for distributed AI.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set, TYPE_CHECKING

# ─────────────────────────────────────────────────────────────────────────────
#  TELEMETRY - OpenTelemetry tracing + Prometheus metrics (Phase 5)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from hololoom.telemetry.tracing import get_tracer, SpanKind
    from hololoom.telemetry.tracing.context import W3CTraceContext
    from hololoom.telemetry.metrics import get_registry
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    get_tracer = None
    SpanKind = None
    W3CTraceContext = None
    get_registry = None

# Federation-specific Prometheus metrics (Phase 5)
try:
    from .metrics import (
        record_federation_query,
        record_query_error,
        record_alignment_check,
        record_alignment_failure,
        set_cluster_nodes,
        inc_pending_queries,
        dec_pending_queries,
    )
    FEDERATION_METRICS_AVAILABLE = True
except ImportError:
    FEDERATION_METRICS_AVAILABLE = False
    record_federation_query = None
    record_query_error = None
    record_alignment_check = None
    record_alignment_failure = None
    set_cluster_nodes = None
    inc_pending_queries = None
    dec_pending_queries = None

if TYPE_CHECKING:
    from hololoom.hololoom import HoloLoom as HoloLoomType

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

# Alignment imports - Federation Alignment Protocol (Phase 0)
from .alignment import (
    # Protocol and types
    AlignmentLayer,
    AlignmentStatus,
    AlignmentVerification,
    ConsensusAlignmentResult,
    # Metadata
    AlignmentMetadata,
    AlignmentProof,
    AlignedResponse,
    ResourceUsage,
    # Verifier
    AlignmentVerifier,
    create_alignment_verifier,
    # Consensus
    AlignmentConsensusEngine,
    create_consensus_engine,
    # Node profiles
    NodeProfileRegistry,
    NodeAlignmentProfile,
    create_profile_registry,
    get_recommended_verification_level,
    TrustLevel,
    # Config
    FederationAlignmentConfig,
    AlignmentMode,
    create_alignment_config,
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

    # ─────────────────────────────────────────────────────────────────────────
    #  ALIGNMENT - Federation Alignment Protocol (Phase 0)
    # ─────────────────────────────────────────────────────────────────────────
    alignment: FederationAlignmentConfig = field(
        default_factory=FederationAlignmentConfig.standard
    )

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
            alignment=FederationAlignmentConfig.permissive(),
        )

    @classmethod
    def production(cls) -> "FederationConfig":
        """Strict settings for production."""
        return cls(
            default_timeout_ms=3000,
            connection_timeout_ms=2000,
            min_trust_score=0.7,
            alignment=FederationAlignmentConfig.standard(),
        )

    @classmethod
    def high_security(cls) -> "FederationConfig":
        """High-security settings with strict alignment verification."""
        return cls(
            default_timeout_ms=3000,
            connection_timeout_ms=2000,
            min_trust_score=0.8,
            alignment=FederationAlignmentConfig.strict(),
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

        # ─────────────────────────────────────────────────────────────────────
        #  ALIGNMENT - Federation Alignment Protocol components
        # ─────────────────────────────────────────────────────────────────────
        self._alignment_verifier: Optional[AlignmentVerifier] = None
        self._consensus_engine: Optional[AlignmentConsensusEngine] = None
        self._node_profiles: Optional[NodeProfileRegistry] = None

        # State
        self._connected = False
        self._peers: Dict[str, FederationNode] = {}
        self._my_guilds: Set[str] = set()
        self._metrics: Dict[str, Any] = {}
        self._started_at: Optional[datetime] = None

        # ─────────────────────────────────────────────────────────────────────
        #  TELEMETRY - Prometheus metrics (Phase 5)
        # ─────────────────────────────────────────────────────────────────────
        self._tracer = None
        self._prom_metrics = None
        if TELEMETRY_AVAILABLE and get_registry is not None:
            try:
                self._tracer = get_tracer("federation")
                registry = get_registry()
                self._prom_metrics = {
                    "gossip_round_duration": registry.histogram(
                        "federation_gossip_round_duration_seconds",
                        "Duration of gossip rounds in seconds",
                        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
                    ),
                    "nodes_discovered": registry.gauge(
                        "federation_nodes_discovered",
                        "Number of nodes discovered in federation",
                    ),
                    "consensus_rounds": registry.counter(
                        "federation_consensus_rounds_total",
                        "Total number of consensus rounds",
                        labels=["algorithm", "outcome"],
                    ),
                    "query_duration": registry.histogram(
                        "federation_query_duration_seconds",
                        "Duration of federated queries",
                        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
                    ),
                    "verification_duration": registry.histogram(
                        "federation_verification_duration_seconds",
                        "Duration of alignment verification",
                        labels=["layer"],
                        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
                    ),
                    "alignment_checks": registry.counter(
                        "federation_alignment_checks_total",
                        "Total alignment checks by layer and result",
                        labels=["layer", "result"],
                    ),
                }
            except Exception as e:
                logger.debug(f"Prometheus metrics unavailable: {e}")

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

        # ─────────────────────────────────────────────────────────────────────
        #  ALIGNMENT - Initialize Federation Alignment Protocol components
        # ─────────────────────────────────────────────────────────────────────
        if self._config.alignment.is_enabled:
            # Create alignment verifier (composes SafetyGuardrails, DeceptionDetector)
            self._alignment_verifier = create_alignment_verifier(
                config=self._config.alignment
            )

            # Create consensus engine for L3 Byzantine agreement
            self._consensus_engine = create_consensus_engine(
                config=self._config.alignment
            )

            # Create node profile registry for trust tracking
            self._node_profiles = create_profile_registry()

            logger.info(
                f"Alignment initialized: mode={self._config.alignment.mode.name}, "
                f"L1={'on' if self._config.alignment.l1.enabled else 'off'}, "
                f"L2={'on' if self._config.alignment.l2.enabled else 'off'}, "
                f"L3={'on' if self._config.alignment.l3.enabled else 'off'}"
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

        # Start OpenTelemetry span for join operation
        span_ctx = None
        if self._tracer and SpanKind:
            span_ctx = self._tracer.trace(
                "federation.join",
                kind=SpanKind.CLIENT,
                attributes={
                    "federation.node_id": self.node_id[:8],
                    "federation.bootstrap": bootstrap,
                },
            )
            span_ctx.__enter__()

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
        finally:
            # End OpenTelemetry span
            if span_ctx:
                span_ctx.__exit__(None, None, None)

    async def leave(self) -> None:
        """Gracefully leave the network."""
        if not self._connected:
            return

        logger.info("Leaving federation...")

        # Start OpenTelemetry span for leave operation
        span_ctx = None
        if self._tracer and SpanKind:
            span_ctx = self._tracer.trace(
                "federation.leave",
                kind=SpanKind.CLIENT,
                attributes={
                    "federation.node_id": self.node_id[:8],
                    "federation.peer_count": len(self._peers),
                },
            )
            span_ctx.__enter__()

        try:
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
        finally:
            # End OpenTelemetry span
            if span_ctx:
                span_ctx.__exit__(None, None, None)

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
        """
        Execute query through federation with alignment verification.

        Flow:
        1. L1 Local Check (<10ms) - Quick safety gate
        2. Route request - Find capable nodes via Kademlia
        3. Collect response - Get response with AlignmentMetadata
        4. L2 Verify Response (10-100ms) - Full alignment check
        5. L3 Consensus (100-500ms) - For high-risk/disputed (optional)
        6. Update Node Profile - Track trust scores
        7. Return aligned response
        """
        import time

        start = time.perf_counter()
        query_result = "unknown"  # Track result for metrics
        query_complexity = "moderate"  # Default complexity

        # Track pending queries for Prometheus metrics
        if FEDERATION_METRICS_AVAILABLE and inc_pending_queries:
            inc_pending_queries()

        # Start OpenTelemetry span for federated query
        span_ctx = None
        if self._tracer and SpanKind:
            span_ctx = self._tracer.trace(
                "federation.federated_query",
                kind=SpanKind.CLIENT,
                attributes={
                    "federation.node_id": self.node_id[:8],
                    "federation.query_text": query.text[:100] if query.text else "",
                    "federation.verification_level": str(query.level) if hasattr(query, 'level') else "STANDARD",
                },
            )
            span_ctx.__enter__()

        try:
            alignment_config = self._config.alignment

            # ─────────────────────────────────────────────────────────────────────
            #  L1: LOCAL CHECK - Quick safety gate before routing
            # ─────────────────────────────────────────────────────────────────────
            if alignment_config.is_enabled and alignment_config.l1.enabled:
                l1_start = time.perf_counter()
                l1_result = await self._l1_local_check(query)
                l1_ms = (time.perf_counter() - l1_start) * 1000

                trace.add("l1_check", self.node_id, l1_ms, passed=l1_result.is_aligned)

                if not l1_result.is_aligned:
                    # Request blocked by L1 check
                    logger.warning(f"L1 blocked query: {l1_result.risk_level}")
                    trace.complete()
                    return FederatedResponse(
                        answer=f"[Query blocked by safety check: {l1_result.risk_level}]",
                        confidence=0.0,
                        verified=False,
                        verified_by=[],
                        trace=trace,
                        source="federation",
                        latency_ms=(time.perf_counter() - start) * 1000,
                        alignment_verification=l1_result,
                    )

            # ─────────────────────────────────────────────────────────────────────
            #  ROUTE: Find capable nodes
            # ─────────────────────────────────────────────────────────────────────
            route_start = time.perf_counter()

            # TODO: Replace with actual Kademlia routing when transport is ready
            # For now, simulate finding nodes
            capable_nodes = list(self._peers.values())[:3]  # Top 3 peers
            route_ms = (time.perf_counter() - route_start) * 1000

            trace.add("route", self.node_id, route_ms, nodes_found=len(capable_nodes))

            if not capable_nodes:
                # No capable nodes found - fall back to local
                logger.warning("No capable nodes found, falling back to local")
                return await self._local_query(query, trace)

            # ─────────────────────────────────────────────────────────────────────
            #  QUERY: Get response from a capable node
            # ─────────────────────────────────────────────────────────────────────
            # TODO: Implement actual transport when ready
            # For now, create a simulated response with alignment metadata
            responder_node = capable_nodes[0] if capable_nodes else None
            responder_id = responder_node.node_id if responder_node else "unknown"

            # Get responder's trust level from profile
            responder_trust = TrustLevel.UNKNOWN
            if self._node_profiles is not None:
                profile = self._node_profiles.get_or_create(responder_id)
                responder_trust = profile.trust_level

            # Simulate response with alignment metadata
            query_ms = 100.0  # Simulated latency
            trace.add("query", responder_id, query_ms)

            # Create alignment metadata (in real impl, this comes from responder)
            alignment_metadata = AlignmentMetadata(
                node_id=responder_id,
                request_id=query.request_id,
                safety_decision_allowed=True,
                risk_level="LOW",
                deception_score=0.1,
                deception_flags=[],
                convergence_safe=True,
                resource_usage=ResourceUsage(cpu_ms=50.0, tokens_out=100),
                reasoning_chain=["Received query", "Retrieved context", "Generated response"],
            )

            # Simulated response
            simulated_answer = f"[Simulated federated response for: {query.text[:50]}...]"
            simulated_confidence = 0.75

            # ─────────────────────────────────────────────────────────────────────
            #  L2: VERIFY RESPONSE - Full alignment check
            # ─────────────────────────────────────────────────────────────────────
            l2_result: Optional[AlignmentVerification] = None
            if alignment_config.is_enabled and alignment_config.l2.enabled:
                l2_result = await self._l2_verify_response(
                    responder_id=responder_id,
                    response_text=simulated_answer,
                    confidence=simulated_confidence,
                    alignment_metadata=alignment_metadata,
                    trace=trace,
                )

                if not l2_result.is_aligned:
                    # Response failed L2 verification
                    # Check if we should escalate to L3 or reject
                    if l2_result.requires_escalation and alignment_config.l3.enabled:
                        # Escalate to L3 consensus
                        pass  # Handled below
                    else:
                        # Reject response
                        logger.warning(
                            f"L2 rejected response from {responder_id}: "
                            f"confidence={l2_result.confidence:.2f}, "
                            f"deception={l2_result.deception_score:.2f}"
                        )
                        # Update node profile with failure
                        await self._update_node_profile(
                            responder_id, l2_result, success=False
                        )

                        trace.complete()
                        query_result = "l2_rejected"  # Track for metrics
                        return FederatedResponse(
                            answer="[Response rejected: failed alignment verification]",
                            confidence=0.0,
                            verified=False,
                            verified_by=[],
                            trace=trace,
                            source="federation",
                            latency_ms=(time.perf_counter() - start) * 1000,
                            alignment_metadata=alignment_metadata,
                            alignment_verification=l2_result,
                            responder_trust_level=responder_trust,
                        )

            # ─────────────────────────────────────────────────────────────────────
            #  L3: CONSENSUS - Byzantine agreement (if needed)
            # ─────────────────────────────────────────────────────────────────────
            l3_result: Optional[ConsensusAlignmentResult] = None
            verified_by: List[str] = []

            needs_l3 = (
                alignment_config.is_enabled
                and alignment_config.l3.enabled
                and (
                    # High-risk responses need consensus
                    (l2_result and l2_result.requires_escalation)
                    # Low-trust nodes need consensus
                    or responder_trust in (TrustLevel.UNTRUSTED, TrustLevel.UNKNOWN)
                    # Critical verification level needs consensus
                    or query.level == VerificationLevel.CRITICAL
                )
            )

            if needs_l3:
                l3_result = await self._l3_consensus_verify(
                    query=query,
                    response_text=simulated_answer,
                    confidence=simulated_confidence,
                    alignment_metadata=alignment_metadata,
                    trace=trace,
                )

                if l3_result.verified:
                    verified_by = l3_result.verifier_nodes
                else:
                    # L3 consensus failed - reject
                    logger.warning(
                        f"L3 consensus failed: "
                        f"agreement={l3_result.agreement_ratio:.2f}, "
                        f"min_required={l3_result.min_required}"
                    )
                    await self._update_node_profile(
                        responder_id, l2_result, success=False
                    )

                    trace.complete()
                    query_result = "l3_rejected"  # Track for metrics
                    return FederatedResponse(
                        answer="[Response rejected: failed consensus verification]",
                        confidence=0.0,
                        verified=False,
                        verified_by=l3_result.verifier_nodes,
                        trace=trace,
                        source="federation",
                        latency_ms=(time.perf_counter() - start) * 1000,
                        alignment_metadata=alignment_metadata,
                        alignment_verification=l2_result,
                        alignment_consensus=l3_result,
                        responder_trust_level=responder_trust,
                    )

            # ─────────────────────────────────────────────────────────────────────
            #  SUCCESS: Update profile and return response
            # ─────────────────────────────────────────────────────────────────────
            await self._update_node_profile(responder_id, l2_result, success=True)

            trace.add("respond", responder_id, 1.0)
            trace.complete()

            elapsed_ms = (time.perf_counter() - start) * 1000
            query_result = "success"  # Track for metrics

            return FederatedResponse(
                answer=simulated_answer,
                confidence=simulated_confidence,
                verified=l3_result.verified if l3_result else (l2_result.is_aligned if l2_result else False),
                verified_by=verified_by,
                trace=trace,
                source="federation",
                latency_ms=elapsed_ms,
                alignment_metadata=alignment_metadata,
                alignment_verification=l2_result,
                alignment_consensus=l3_result,
                responder_trust_level=responder_trust,
            )
        finally:
            # Record federated query metrics
            if FEDERATION_METRICS_AVAILABLE and record_federation_query:
                elapsed_seconds = time.perf_counter() - start
                # Use context manager protocol to record query metrics
                try:
                    from .metrics import QueryMetricsContext
                    ctx = QueryMetricsContext(complexity=query_complexity)
                    ctx.set_result(query_result)
                    # Manually record since we can't use context manager here
                    from .metrics import (
                        _init_metrics,
                        _federation_queries_total,
                        _federation_query_latency,
                    )
                    _init_metrics()
                    if _federation_queries_total:
                        _federation_queries_total.inc(
                            labels={"result": query_result, "complexity": query_complexity}
                        )
                    if _federation_query_latency:
                        _federation_query_latency.observe(
                            elapsed_seconds,
                            labels={"result": query_result},
                        )
                except Exception:
                    pass  # Metrics are best-effort

            # Decrement pending queries for Prometheus metrics
            if FEDERATION_METRICS_AVAILABLE and dec_pending_queries:
                dec_pending_queries()

            # End OpenTelemetry span
            if span_ctx:
                span_ctx.__exit__(None, None, None)

    # ───────────────────────────────────────────────────────────────────────
    #  ALIGNMENT HELPERS - L1/L2/L3 verification methods
    # ───────────────────────────────────────────────────────────────────────

    async def _l1_local_check(
        self,
        query: Query,
    ) -> AlignmentVerification:
        """
        L1: Quick local safety check (<10ms target).

        Checks if the query can be safely processed before routing.
        """
        import time
        start = time.perf_counter()

        # Start OpenTelemetry span for L1 check
        span_ctx = None
        if self._tracer and SpanKind:
            span_ctx = self._tracer.trace(
                "federation.l1_local_check",
                kind=SpanKind.INTERNAL,
                attributes={
                    "federation.node_id": self.node_id[:8],
                    "federation.query_text": query.text[:100] if query.text else "",
                },
            )
            span_ctx.__enter__()

        try:
            if self._alignment_verifier is None:
                # No verifier - return default pass
                latency_s = time.perf_counter() - start
                # Record L1 alignment check metric
                if FEDERATION_METRICS_AVAILABLE and record_alignment_check:
                    record_alignment_check(
                        layer="L1_LOCAL",
                        result="aligned",
                        latency_seconds=latency_s,
                    )
                return AlignmentVerification(
                    node_id=self.node_id,
                    request_id=query.request_id,
                    status=AlignmentStatus.VERIFIED,
                    layer=AlignmentLayer.L1_LOCAL,
                    confidence=1.0,
                    risk_level="SAFE",
                    deception_score=0.0,
                    latency_ms=latency_s * 1000,
                )

            # Use alignment verifier for L1 check
            # Create a minimal Response object for verification
            response = Response(
                text=query.text,
                request_id=query.request_id,
                responder=self.node_id,
                confidence=1.0,
            )

            verification = await self._alignment_verifier.verify_node_alignment(
                node_id=self.node_id,
                response=response,
                alignment_threshold=self._config.alignment.l2.alignment_threshold,
                layer=AlignmentLayer.L1_LOCAL,
            )

            # Record L1 alignment check metric
            if FEDERATION_METRICS_AVAILABLE and record_alignment_check:
                result = "aligned" if verification.is_aligned else "misaligned"
                record_alignment_check(
                    layer="L1_LOCAL",
                    result=result,
                    latency_seconds=verification.latency_ms / 1000,
                )
                # Record failure details if misaligned
                if not verification.is_aligned and record_alignment_failure:
                    record_alignment_failure(
                        layer="L1_LOCAL",
                        reason=verification.risk_level or "unknown",
                    )

            return verification
        finally:
            # End OpenTelemetry span
            if span_ctx:
                span_ctx.__exit__(None, None, None)

    async def _l2_verify_response(
        self,
        responder_id: str,
        response_text: str,
        confidence: float,
        alignment_metadata: AlignmentMetadata,
        trace: QueryTrace,
    ) -> AlignmentVerification:
        """
        L2: Full response verification (10-100ms target).

        Verifies alignment metadata and runs deception detection.
        """
        import time
        start = time.perf_counter()

        # Start OpenTelemetry span for L2 verification
        span_ctx = None
        if self._tracer and SpanKind:
            span_ctx = self._tracer.trace(
                "federation.l2_verify_response",
                kind=SpanKind.INTERNAL,
                attributes={
                    "federation.node_id": self.node_id[:8],
                    "federation.responder_id": responder_id[:8],
                    "federation.confidence": confidence,
                },
            )
            span_ctx.__enter__()

        try:
            if self._alignment_verifier is None:
                # No verifier - return based on metadata alone
                result = AlignmentVerification(
                    node_id=responder_id,
                    request_id=alignment_metadata.request_id,
                    status=AlignmentStatus.VERIFIED if alignment_metadata.is_aligned else AlignmentStatus.FAILED,
                    layer=AlignmentLayer.L2_RESPONSE,
                    confidence=alignment_metadata.alignment_confidence,
                    risk_level=alignment_metadata.risk_level,
                    deception_score=alignment_metadata.deception_score,
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
                trace.add("l2_verify", self.node_id, result.latency_ms, passed=result.is_aligned)
                # Record L2 alignment check metric
                if FEDERATION_METRICS_AVAILABLE and record_alignment_check:
                    metric_result = "aligned" if result.is_aligned else "misaligned"
                    record_alignment_check(
                        layer="L2_RESPONSE",
                        result=metric_result,
                        latency_seconds=result.latency_ms / 1000,
                    )
                    if not result.is_aligned and record_alignment_failure:
                        record_alignment_failure(
                            layer="L2_RESPONSE",
                            reason=result.risk_level or "unknown",
                        )
                return result

            # Create Response object for verification
            response = Response(
                text=response_text,
                request_id=alignment_metadata.request_id,
                responder=responder_id,
                confidence=confidence,
            )

            verification = await self._alignment_verifier.verify_node_alignment(
                node_id=responder_id,
                response=response,
                alignment_threshold=self._config.alignment.l2.alignment_threshold,
                layer=AlignmentLayer.L2_RESPONSE,
            )

            trace.add("l2_verify", self.node_id, verification.latency_ms, passed=verification.is_aligned)
            # Record L2 alignment check metric
            if FEDERATION_METRICS_AVAILABLE and record_alignment_check:
                metric_result = "aligned" if verification.is_aligned else "misaligned"
                record_alignment_check(
                    layer="L2_RESPONSE",
                    result=metric_result,
                    latency_seconds=verification.latency_ms / 1000,
                )
                if not verification.is_aligned and record_alignment_failure:
                    record_alignment_failure(
                        layer="L2_RESPONSE",
                        reason=verification.risk_level or "unknown",
                    )
            return verification
        finally:
            # End OpenTelemetry span
            if span_ctx:
                span_ctx.__exit__(None, None, None)

    async def _l3_consensus_verify(
        self,
        query: Query,
        response_text: str,
        confidence: float,
        alignment_metadata: AlignmentMetadata,
        trace: QueryTrace,
    ) -> ConsensusAlignmentResult:
        """
        L3: Multi-node Byzantine consensus (100-500ms target).

        Used for high-risk responses or disputed verifications.
        """
        import time
        start = time.perf_counter()

        # Start OpenTelemetry span for L3 consensus
        span_ctx = None
        if self._tracer and SpanKind:
            span_ctx = self._tracer.trace(
                "federation.l3_consensus_verify",
                kind=SpanKind.INTERNAL,
                attributes={
                    "federation.node_id": self.node_id[:8],
                    "federation.query_id": query.request_id[:8] if query.request_id else "",
                    "federation.confidence": confidence,
                },
            )
            span_ctx.__enter__()

        try:
            if self._consensus_engine is None:
                # No consensus engine - return synthetic result
                result = ConsensusAlignmentResult(
                    request_id=query.request_id,
                    verified=alignment_metadata.is_aligned,
                    agreement_ratio=1.0 if alignment_metadata.is_aligned else 0.0,
                    min_required=1,
                    verifier_nodes=[self.node_id],
                    consensus_confidence=alignment_metadata.alignment_confidence,
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
                trace.add("l3_consensus", self.node_id, result.latency_ms, agreed=result.verified)
                # Record L3 alignment check metric
                if FEDERATION_METRICS_AVAILABLE and record_alignment_check:
                    metric_result = "aligned" if result.verified else "misaligned"
                    record_alignment_check(
                        layer="L3_CONSENSUS",
                        result=metric_result,
                        latency_seconds=result.latency_ms / 1000,
                    )
                    if not result.verified and record_alignment_failure:
                        record_alignment_failure(
                            layer="L3_CONSENSUS",
                            reason="consensus_failed",
                        )
                return result

            # Get verifier nodes (use peers with good trust)
            verifier_nodes = []
            if self._node_profiles is not None:
                for peer_id in self._peers.keys():
                    profile = self._node_profiles.get_or_create(peer_id)
                    if profile.trust_level in (TrustLevel.TRUSTED, TrustLevel.HIGHLY_TRUSTED):
                        verifier_nodes.append(peer_id)
                        if len(verifier_nodes) >= self._config.alignment.l3.max_verifiers:
                            break

            # Need minimum verifiers
            min_verifiers = self._config.alignment.l3.min_verifiers
            if len(verifier_nodes) < min_verifiers:
                # Not enough trusted verifiers - use what we have
                verifier_nodes = list(self._peers.keys())[:min_verifiers]

            # Create Response object for consensus
            response = Response(
                text=response_text,
                request_id=query.request_id,
                responder=alignment_metadata.node_id,
                confidence=confidence,
            )

            consensus = await self._consensus_engine.consensus_alignment_verify(
                response=response,
                verifier_nodes=verifier_nodes,
                min_agreements=min_verifiers,
            )

            trace.add("l3_consensus", self.node_id, consensus.latency_ms, agreed=consensus.verified)
            # Record L3 alignment check metric
            if FEDERATION_METRICS_AVAILABLE and record_alignment_check:
                metric_result = "aligned" if consensus.verified else "misaligned"
                record_alignment_check(
                    layer="L3_CONSENSUS",
                    result=metric_result,
                    latency_seconds=consensus.latency_ms / 1000,
                )
                if not consensus.verified and record_alignment_failure:
                    record_alignment_failure(
                        layer="L3_CONSENSUS",
                        reason=f"agreement_ratio_{consensus.agreement_ratio:.2f}",
                    )
            return consensus
        finally:
            # End OpenTelemetry span
            if span_ctx:
                span_ctx.__exit__(None, None, None)

    async def _update_node_profile(
        self,
        node_id: str,
        verification: Optional[AlignmentVerification],
        success: bool,
    ) -> None:
        """Update node trust profile based on verification outcome."""
        if self._node_profiles is None:
            return

        profile = self._node_profiles.get_or_create(node_id)

        if success:
            # Verified response - record success
            profile.record_verified_response(
                verification.metadata.get("request_id", "") if verification else ""
            )
        else:
            # Failed verification - record incident
            if verification and verification.deception_score > 0.5:
                # Deception detected
                from .alignment.node_profile import AlignmentEventType
                profile.record_event(
                    AlignmentEventType.DECEPTION_DETECTED,
                    details={
                        "deception_score": verification.deception_score,
                        "flags": [f.value for f in verification.deception_flags],
                    },
                )
            else:
                # Resource violation or other failure
                from .alignment.node_profile import AlignmentEventType
                profile.record_event(
                    AlignmentEventType.RESOURCE_VIOLATION,
                    details={"reason": "verification_failed"},
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

    Contains the answer plus complete provenance and alignment metadata.

    Every federated response carries alignment proof:
    - alignment_metadata: AlignmentMetadata from the responding node
    - alignment_verification: AlignmentVerification from L2 check
    - alignment_consensus: ConsensusAlignmentResult from L3 (if escalated)
    """

    answer: str
    confidence: float                         # 0.0-1.0
    verified: bool                            # Whether consensus was reached
    verified_by: List[str]                    # Node IDs that verified
    trace: QueryTrace                         # Full execution trace
    source: str                               # "federation" or "local"
    latency_ms: float                         # Total latency

    # ─────────────────────────────────────────────────────────────────────────
    #  ALIGNMENT - Federation Alignment Protocol metadata
    # ─────────────────────────────────────────────────────────────────────────
    alignment_metadata: Optional[AlignmentMetadata] = None        # From responder
    alignment_verification: Optional[AlignmentVerification] = None  # L2 result
    alignment_consensus: Optional[ConsensusAlignmentResult] = None  # L3 result (if escalated)
    responder_trust_level: Optional[TrustLevel] = None            # Responder's trust level

    @property
    def is_aligned(self) -> bool:
        """Check if response passed alignment verification."""
        if self.alignment_verification is not None:
            return self.alignment_verification.is_aligned
        if self.alignment_metadata is not None:
            return self.alignment_metadata.is_aligned
        return False  # No alignment data = not aligned

    @property
    def alignment_confidence(self) -> float:
        """Get alignment confidence (0.0-1.0)."""
        if self.alignment_verification is not None:
            return self.alignment_verification.confidence
        if self.alignment_metadata is not None:
            return self.alignment_metadata.alignment_confidence
        return 0.0

    def __str__(self) -> str:
        status = "✓" if self.verified else "○"
        align_status = "🛡" if self.is_aligned else "⚠"
        return f"{status}{align_status} ({self.confidence:.0%}) {self.answer[:100]}..."

    def __repr__(self) -> str:
        return (
            f"FederatedResponse("
            f"verified={self.verified}, "
            f"aligned={self.is_aligned}, "
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
