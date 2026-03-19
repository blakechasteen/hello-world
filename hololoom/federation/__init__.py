"""
HoloLoom Federation - Decentralized AI Safety Network.

async with Federation() as fed:
    await fed.join("bootstrap.hololoom.net:9000")
    result = await fed.query("What is love?")

That's it. That's the API.

For more control:
    result = await fed.query(
        "Is this safe?",
        level=VerificationLevel.CRITICAL,
        guild="medical"
    )
"""

from __future__ import annotations

from .agentic_rag import (
    # Constants
    DEFAULT_FEDERATION_THRESHOLD,
    RAG_RECALL_METHOD,
    # Main classes
    ConfidenceAggregator,
    FederatedRAG,
    # Data classes
    FederatedRAGConfig,
    FederatedRAGResult,
    # Factory
    create_federated_rag,
)

# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC API - What users should import
# ═══════════════════════════════════════════════════════════════════════════
# The one class you need
from .core import (
    FederatedResponse,
    Federation,
    FederationConfig,
    connect,
)

# Gossip membership
from .gossip import (
    MessageType,
    SwimMembership,
)

# Guild management
from .guild import (
    GuildManager,
    TrustCalculator,
)

# ═══════════════════════════════════════════════════════════════════════════
#  ADVANCED API - For power users
# ═══════════════════════════════════════════════════════════════════════════
# Identity management
from .identity import (
    Identity,
    create_node,
    get_or_create_identity,
)
from .inference_router import (
    DEFAULT_INFERENCE_TIMEOUT_SECONDS,
    DEFAULT_STREAM_TIMEOUT_SECONDS,
    INFERENCE_CAPABILITIES_METHOD,
    # Constants
    INFERENCE_GENERATE_METHOD,
    # Data classes
    InferenceRequest,
    InferenceResult,
    InferenceRouter,
    # Enums
    InferenceStatus,
    InferenceToken,
    NodeCapabilities,
    # Classes
    StreamingProxy,
    # Factory
    create_inference_router,
)

# Distributed Inference (Day 19)
from .load_balancer import (
    DEFAULT_LATENCY_WEIGHT,
    # Constants
    DEFAULT_LOAD_WEIGHT,
    DEFAULT_SUCCESS_WEIGHT,
    DEFAULT_TRUST_WEIGHT,
    MAX_CONNECTIONS_PER_NODE,
    OVERLOAD_THRESHOLD,
    # Classes
    CircuitBreaker,
    CircuitBreakerState,
    LoadBalancer,
    LoadBalancerConfig,
    # Enums
    LoadBalanceStrategy,
    NodeHealth,
    # Data classes
    NodeStats,
    SelectionResult,
    # Factory
    create_load_balancer,
)

# ═══════════════════════════════════════════════════════════════════════════
#  OBSERVABILITY - Phase 5 (December 2025)
# ═══════════════════════════════════════════════════════════════════════════
# Prometheus metrics
from .metrics import (
    # Collector
    FederationCollector,
    FederationMetricsSnapshot,
    QueryMetricsContext,
    dec_pending_queries,
    get_federation_collector,
    inc_pending_queries,
    # Alignment metrics
    record_alignment_check,
    record_alignment_failure,
    # Query metrics
    record_federation_query,
    # Gossip metrics
    record_gossip_message,
    record_membership_change,
    record_query_error,
    # Cluster metrics
    set_cluster_nodes,
    set_pending_queries,
)

# Rate limiting (Day 15)
from .rate_limiter import (
    DEFAULT_RATE_LIMITS,
    TRUST_TO_TIER,
    FederatedRateLimiter,
    RateLimitInfo,
    RateLimitState,
    RateLimitTier,
    create_rate_limiter,
    get_tier_for_trust_level,
)

# Agentic RAG (Day 18)
from .result_merger import (
    DEFAULT_SIMILARITY_THRESHOLD,
    MIN_MI_THRESHOLD,
    # Constants
    TRUST_WEIGHTS,
    MergedRAGResult,
    NodeRAGResult,
    # Main class
    RAGResultMerger,
    # Data classes
    SourceWithProvenance,
    # Factory
    create_result_merger,
)

# DHT routing
from .routing import (
    KademliaRouter,
    RoutingTable,
)

# ═══════════════════════════════════════════════════════════════════════════
#  SAFETY & SECURITY - Phase 4 (December 2025)
# ═══════════════════════════════════════════════════════════════════════════
# Safety layer (Day 15)
from .safety import (
    METHOD_PERMISSIONS,
    TRUST_PERMISSIONS,
    FederationPermission,
    FederationSafetyGate,
    FederationSafetyResult,
    GuildTrustChecker,
    SafetyCheckResult,
    SignatureVerifier,
    SignedRequest,
    create_federation_safety_gate,
    parse_signed_request,
)
from .transport.matrix_room import (
    # Constants (also exported from matrix_transport, but aliased here)
    HOLOLOOM_CAPABILITIES_STATE,
    # Data classes
    AgentCapabilities,
    # Room abstraction
    MatrixAgentRoom,
    RoomCapabilitySummary,
    # Factory
    create_agent_room,
)

# Matrix transport (Day 17)
from .transport.matrix_transport import (
    HAS_MATRIX_NIO,
    HOLOLOOM_CAPABILITIES_TYPE,
    # Constants
    HOLOLOOM_RPC_MSGTYPE,
    MatrixRPCEvent,
    MatrixTransportAdapter,
    # Classes
    MatrixTrustResolver,
    create_matrix_transport,
    create_matrix_trust_resolver,
    # Functions
    matrix_user_to_node_id,
    parse_matrix_rpc_event,
)

# Types you'll use
from .types import (
    AdmissionPolicy,
    # Capabilities
    Capability,
    # Errors (for catching)
    FederationError,
    # Nodes and Guilds
    FederationNode,
    Guild,
    GuildError,
    GuildTrustLevel,
    NetworkError,
    # Query/Response
    Query,
    QueryTrace,
    Response,
    RoutingError,
    TimeoutError,
    Verification,
    VerificationError,
    # Verification
    VerificationLevel,
)

# Consensus verification
from .verifier import (
    ConsensusVerifier,
    DSStarScorer,
    VerificationScore,
    get_quorum,
)

# Wire protocol (Day 16)
from .wire_protocol import (
    ERROR_MESSAGES,
    HOLOLOOM_METHODS,
    ErrorCode,
    JSONRPCBuilder,
    RequestMeta,
    RequestValidator,
    RPCError,
    RPCRequest,
    RPCResponse,
    ValidationResult,
    create_builder,
    create_validator,
    parse_batch,
    parse_request,
)

# ═══════════════════════════════════════════════════════════════════════════
#  VERSION
# ═══════════════════════════════════════════════════════════════════════════

__version__ = "0.1.0"

__all__ = [
    # Core (what 90% of users need)
    "Federation",
    "FederationConfig",
    "FederatedResponse",
    "connect",
    # Types
    "VerificationLevel",
    "Verification",
    "Capability",
    "Query",
    "Response",
    "QueryTrace",
    "FederationNode",
    "Guild",
    "GuildTrustLevel",
    "AdmissionPolicy",
    # Errors
    "FederationError",
    "NetworkError",
    "RoutingError",
    "TimeoutError",
    "VerificationError",
    "GuildError",
    # Advanced - Identity & Routing
    "Identity",
    "create_node",
    "get_or_create_identity",
    "GuildManager",
    "TrustCalculator",
    "ConsensusVerifier",
    "DSStarScorer",
    "VerificationScore",
    "get_quorum",
    "KademliaRouter",
    "RoutingTable",
    "SwimMembership",
    "MessageType",
    # Safety & Security (Phase 4 - December 2025)
    "FederationSafetyGate",
    "SignatureVerifier",
    "GuildTrustChecker",
    "SafetyCheckResult",
    "SignedRequest",
    "FederationSafetyResult",
    "FederationPermission",
    "create_federation_safety_gate",
    "parse_signed_request",
    "TRUST_PERMISSIONS",
    "METHOD_PERMISSIONS",
    # Rate Limiting
    "FederatedRateLimiter",
    "RateLimitTier",
    "RateLimitState",
    "RateLimitInfo",
    "get_tier_for_trust_level",
    "create_rate_limiter",
    "DEFAULT_RATE_LIMITS",
    "TRUST_TO_TIER",
    # Wire Protocol (JSON-RPC 2.0)
    "JSONRPCBuilder",
    "RequestValidator",
    "RPCRequest",
    "RPCResponse",
    "RPCError",
    "RequestMeta",
    "ErrorCode",
    "ValidationResult",
    "HOLOLOOM_METHODS",
    "ERROR_MESSAGES",
    "create_builder",
    "create_validator",
    "parse_request",
    "parse_batch",
    # Matrix Transport (Day 17)
    "HOLOLOOM_RPC_MSGTYPE",
    "HOLOLOOM_CAPABILITIES_TYPE",
    "HOLOLOOM_CAPABILITIES_STATE",
    "HAS_MATRIX_NIO",
    "MatrixTrustResolver",
    "MatrixTransportAdapter",
    "MatrixRPCEvent",
    "matrix_user_to_node_id",
    "parse_matrix_rpc_event",
    "create_matrix_transport",
    "create_matrix_trust_resolver",
    "AgentCapabilities",
    "RoomCapabilitySummary",
    "MatrixAgentRoom",
    "create_agent_room",
    # Agentic RAG (Day 18)
    "SourceWithProvenance",
    "NodeRAGResult",
    "MergedRAGResult",
    "RAGResultMerger",
    "create_result_merger",
    "TRUST_WEIGHTS",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "MIN_MI_THRESHOLD",
    "FederatedRAGConfig",
    "FederatedRAGResult",
    "ConfidenceAggregator",
    "FederatedRAG",
    "create_federated_rag",
    "DEFAULT_FEDERATION_THRESHOLD",
    "RAG_RECALL_METHOD",
    # Distributed Inference (Day 19)
    # Load Balancer
    "LoadBalanceStrategy",
    "NodeHealth",
    "NodeStats",
    "LoadBalancerConfig",
    "SelectionResult",
    "CircuitBreakerState",
    "CircuitBreaker",
    "LoadBalancer",
    "create_load_balancer",
    "DEFAULT_LOAD_WEIGHT",
    "DEFAULT_TRUST_WEIGHT",
    "DEFAULT_LATENCY_WEIGHT",
    "DEFAULT_SUCCESS_WEIGHT",
    "OVERLOAD_THRESHOLD",
    "MAX_CONNECTIONS_PER_NODE",
    # Inference Router
    "InferenceStatus",
    "InferenceRequest",
    "InferenceToken",
    "InferenceResult",
    "NodeCapabilities",
    "StreamingProxy",
    "InferenceRouter",
    "create_inference_router",
    "INFERENCE_GENERATE_METHOD",
    "INFERENCE_CAPABILITIES_METHOD",
    "DEFAULT_INFERENCE_TIMEOUT_SECONDS",
    "DEFAULT_STREAM_TIMEOUT_SECONDS",
    # Observability (Phase 5 - December 2025)
    # Query metrics
    "record_federation_query",
    "record_query_error",
    "QueryMetricsContext",
    # Alignment metrics
    "record_alignment_check",
    "record_alignment_failure",
    # Gossip metrics
    "record_gossip_message",
    "record_membership_change",
    # Cluster metrics
    "set_cluster_nodes",
    "set_pending_queries",
    "inc_pending_queries",
    "dec_pending_queries",
    # Collector
    "FederationCollector",
    "FederationMetricsSnapshot",
    "get_federation_collector",
]
