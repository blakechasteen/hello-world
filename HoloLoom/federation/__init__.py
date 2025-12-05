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
    # Advanced
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
]
