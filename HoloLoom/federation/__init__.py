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

# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC API - What users should import
# ═══════════════════════════════════════════════════════════════════════════

# The one class you need
from .core import (
    Federation,
    FederationConfig,
    FederatedResponse,
    connect,
)

# Types you'll use
from .types import (
    # Verification
    VerificationLevel,
    Verification,
    # Capabilities
    Capability,
    # Query/Response
    Query,
    Response,
    QueryTrace,
    # Nodes and Guilds
    FederationNode,
    Guild,
    GuildTrustLevel,
    AdmissionPolicy,
    # Errors (for catching)
    FederationError,
    NetworkError,
    RoutingError,
    TimeoutError,
    VerificationError,
    GuildError,
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

# Guild management
from .guild import (
    GuildManager,
    TrustCalculator,
)

# Consensus verification
from .consensus import (
    ConsensusVerifier,
    DSStarScorer,
    VerificationScore,
    get_quorum,
)

# DHT routing
from .routing import (
    KademliaRouter,
    RoutingTable,
)

# Gossip membership
from .gossip import (
    SwimMembership,
    MessageType,
)

# ═══════════════════════════════════════════════════════════════════════════
#  SAFETY & SECURITY - Phase 4 (December 2025)
# ═══════════════════════════════════════════════════════════════════════════

# Safety layer (Day 15)
from .safety import (
    FederationSafetyGate,
    SignatureVerifier,
    GuildTrustChecker,
    SafetyCheckResult,
    SignedRequest,
    FederationSafetyResult,
    FederationPermission,
    create_federation_safety_gate,
    parse_signed_request,
    TRUST_PERMISSIONS,
    METHOD_PERMISSIONS,
)

# Rate limiting (Day 15)
from .rate_limiter import (
    FederatedRateLimiter,
    RateLimitTier,
    RateLimitState,
    RateLimitInfo,
    get_tier_for_trust_level,
    create_rate_limiter,
    DEFAULT_RATE_LIMITS,
    TRUST_TO_TIER,
)

# Wire protocol (Day 16)
from .wire_protocol import (
    JSONRPCBuilder,
    RequestValidator,
    RPCRequest,
    RPCResponse,
    RPCError,
    RequestMeta,
    ErrorCode,
    ValidationResult,
    HOLOLOOM_METHODS,
    ERROR_MESSAGES,
    create_builder,
    create_validator,
    parse_request,
    parse_batch,
)

# Matrix transport (Day 17)
from .transport.matrix_transport import (
    # Constants
    HOLOLOOM_RPC_MSGTYPE,
    HOLOLOOM_CAPABILITIES_TYPE,
    HAS_MATRIX_NIO,
    # Classes
    MatrixTrustResolver,
    MatrixTransportAdapter,
    MatrixRPCEvent,
    # Functions
    matrix_user_to_node_id,
    parse_matrix_rpc_event,
    create_matrix_transport,
    create_matrix_trust_resolver,
)

from .transport.matrix_room import (
    # Data classes
    AgentCapabilities,
    RoomCapabilitySummary,
    # Room abstraction
    MatrixAgentRoom,
    # Factory
    create_agent_room,
    # Constants (also exported from matrix_transport, but aliased here)
    HOLOLOOM_CAPABILITIES_STATE,
)

# Agentic RAG (Day 18)
from .result_merger import (
    # Data classes
    SourceWithProvenance,
    NodeRAGResult,
    MergedRAGResult,
    # Main class
    RAGResultMerger,
    # Factory
    create_result_merger,
    # Constants
    TRUST_WEIGHTS,
    DEFAULT_SIMILARITY_THRESHOLD,
    MIN_MI_THRESHOLD,
)

from .agentic_rag import (
    # Data classes
    FederatedRAGConfig,
    FederatedRAGResult,
    # Main classes
    ConfidenceAggregator,
    FederatedRAG,
    # Factory
    create_federated_rag,
    # Constants
    DEFAULT_FEDERATION_THRESHOLD,
    RAG_RECALL_METHOD,
)

# Distributed Inference (Day 19)
from .load_balancer import (
    # Enums
    LoadBalanceStrategy,
    NodeHealth,
    # Data classes
    NodeStats,
    LoadBalancerConfig,
    SelectionResult,
    CircuitBreakerState,
    # Classes
    CircuitBreaker,
    LoadBalancer,
    # Factory
    create_load_balancer,
    # Constants
    DEFAULT_LOAD_WEIGHT,
    DEFAULT_TRUST_WEIGHT,
    DEFAULT_LATENCY_WEIGHT,
    DEFAULT_SUCCESS_WEIGHT,
    OVERLOAD_THRESHOLD,
    MAX_CONNECTIONS_PER_NODE,
)

from .inference_router import (
    # Enums
    InferenceStatus,
    # Data classes
    InferenceRequest,
    InferenceToken,
    InferenceResult,
    NodeCapabilities,
    # Classes
    StreamingProxy,
    InferenceRouter,
    # Factory
    create_inference_router,
    # Constants
    INFERENCE_GENERATE_METHOD,
    INFERENCE_CAPABILITIES_METHOD,
    DEFAULT_INFERENCE_TIMEOUT_SECONDS,
    DEFAULT_STREAM_TIMEOUT_SECONDS,
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
]
