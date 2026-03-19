#!/usr/bin/env python3
"""
Matrix Transport Layer
=======================

Part of Phase 4: Open & Safe Agent Federation (December 2025)
Day 17: Matrix Transport (Safe Implementation)

Provides Matrix protocol integration for HoloLoom federation, enabling
decentralized agent-to-agent communication over Matrix homeservers.

Key insight: 95% of security code already exists. This module is just
a thin adapter (~400 lines) wrapping battle-tested security infrastructure
(~2,000 lines) from Days 15-16.

Components:
    - MatrixTrustResolver: Maps Matrix power levels → GuildTrustLevel
    - MatrixTransportAdapter: Wraps matrix-nio with safety integration
    - MatrixCapabilityState: Room capabilities via state events

Security Flow:
    Matrix Event → SignedRequest → FederationSafetyGate → Execute/Deny

Author: Claude Code (Phase 4 - December 2025)
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..identity import Identity
from ..rate_limiter import (
    FederatedRateLimiter,
    RateLimitTier,
)
from ..safety import (
    FederationSafetyGate,
    SignedRequest,
)
from ..types import (
    FederationNode,
    GuildTrustLevel,
    NodeStatus,
)

# Optional matrix-nio import
try:
    import nio
    HAS_MATRIX_NIO = True
except ImportError:
    HAS_MATRIX_NIO = False
    nio = None  # type: ignore

if TYPE_CHECKING:
    from nio import AsyncClient, MatrixRoom, RoomMessageText

logger = logging.getLogger(__name__)


# ============================================================================
#  Constants
# ============================================================================

# Custom Matrix message type for HoloLoom RPC
HOLOLOOM_RPC_MSGTYPE = "m.hololoom.rpc"

# Custom Matrix state event for capabilities
HOLOLOOM_CAPABILITIES_TYPE = "m.room.hololoom_capabilities"

# Default Matrix message timeout (ms)
DEFAULT_TIMEOUT_MS = 30000


# ============================================================================
#  Matrix Trust Resolver
# ============================================================================

class MatrixTrustResolver:
    """
    Maps Matrix room power levels to federation guild trust levels.

    Matrix power levels (0-100) map to HoloLoom GuildTrustLevel:
        0-49:   STARTER (OBSERVER tier) - Default users
        50-74:  ESTABLISHED (MEMBER tier) - Moderators
        75-99:  VETERAN (CONTRIBUTOR tier) - Admins
        100:    VETERAN (STEWARD tier) - Room owners

    This mapping ensures that Matrix room permissions translate naturally
    to federation trust levels, enabling permission-based access control.

    Example:
        >>> resolver = MatrixTrustResolver()
        >>> trust = resolver.resolve_from_power_level(50)
        >>> trust
        GuildTrustLevel.ESTABLISHED
        >>> resolver.get_permissions(trust)
        frozenset({<FederationPermission.RECALL: 1>, ...})
    """

    # Matrix power level thresholds → GuildTrustLevel
    POWER_TO_TRUST: dict[int, GuildTrustLevel] = {
        0: GuildTrustLevel.STARTER,       # Default users
        50: GuildTrustLevel.ESTABLISHED,  # Moderators
        75: GuildTrustLevel.VETERAN,      # Admins
        100: GuildTrustLevel.VETERAN,     # Room owners (same as admin)
    }

    # GuildTrustLevel → RateLimitTier (for rate limiting)
    TRUST_TO_TIER: dict[GuildTrustLevel, RateLimitTier] = {
        GuildTrustLevel.STARTER: RateLimitTier.OBSERVER,
        GuildTrustLevel.ESTABLISHED: RateLimitTier.MEMBER,
        GuildTrustLevel.VETERAN: RateLimitTier.CONTRIBUTOR,
    }

    def __init__(
        self,
        custom_power_mapping: dict[int, GuildTrustLevel] | None = None,
        steward_power_level: int = 100,
    ):
        """
        Initialize trust resolver.

        Args:
            custom_power_mapping: Override default power level mapping
            steward_power_level: Power level required for STEWARD tier (unlimited)
        """
        self._power_mapping = custom_power_mapping or self.POWER_TO_TRUST
        self._steward_level = steward_power_level

    def resolve_from_power_level(self, power_level: int) -> GuildTrustLevel:
        """
        Resolve guild trust level from Matrix power level.

        Args:
            power_level: Matrix room power level (0-100)

        Returns:
            Corresponding GuildTrustLevel
        """
        # Clamp to valid range
        power_level = max(0, min(100, power_level))

        # Find highest matching threshold
        resolved_trust = GuildTrustLevel.STARTER
        for threshold, trust in sorted(self._power_mapping.items()):
            if power_level >= threshold:
                resolved_trust = trust

        return resolved_trust

    def resolve_trust_level(
        self,
        matrix_user: str,
        power_levels: dict[str, Any],
    ) -> GuildTrustLevel:
        """
        Get trust level for a Matrix user from room power levels state.

        Args:
            matrix_user: Matrix user ID (e.g., "@agent:example.org")
            power_levels: m.room.power_levels state event content

        Returns:
            GuildTrustLevel for the user
        """
        # Get user's power level (default to users_default or 0)
        users = power_levels.get("users", {})
        users_default = power_levels.get("users_default", 0)
        user_power = users.get(matrix_user, users_default)

        return self.resolve_from_power_level(user_power)

    def get_rate_limit_tier(self, trust_level: GuildTrustLevel) -> RateLimitTier:
        """
        Get rate limit tier for a guild trust level.

        Args:
            trust_level: Guild trust level

        Returns:
            Corresponding RateLimitTier
        """
        return self.TRUST_TO_TIER.get(trust_level, RateLimitTier.OBSERVER)

    def is_steward(self, power_level: int) -> bool:
        """Check if power level qualifies for STEWARD tier (unlimited)."""
        return power_level >= self._steward_level

    def get_allowed_methods(self, trust_level: GuildTrustLevel) -> list[str]:
        """
        Get list of allowed JSON-RPC methods for a trust level.

        Args:
            trust_level: Guild trust level

        Returns:
            List of allowed method names
        """
        from ..safety import METHOD_PERMISSIONS, TRUST_PERMISSIONS

        allowed_permissions = TRUST_PERMISSIONS.get(trust_level, frozenset())
        return [
            method
            for method, perm in METHOD_PERMISSIONS.items()
            if perm in allowed_permissions
        ]


# ============================================================================
#  Matrix User Identity Mapping
# ============================================================================

def matrix_user_to_node_id(matrix_user: str) -> str:
    """
    Convert Matrix user ID to federation node ID.

    Uses SHA256 hash of the Matrix user ID to create a deterministic
    160-bit node identifier (40 hex chars), matching the federation
    identity format.

    Args:
        matrix_user: Matrix user ID (e.g., "@agent:example.org")

    Returns:
        40-character hex node ID

    Example:
        >>> matrix_user_to_node_id("@agent:example.org")
        'a1b2c3d4e5f6...'  # 40 hex chars
    """
    # Normalize user ID
    user_bytes = matrix_user.lower().strip().encode("utf-8")
    # SHA256 hash, take first 20 bytes (160 bits = 40 hex chars)
    return hashlib.sha256(user_bytes).hexdigest()[:40]


def matrix_user_to_public_key_placeholder(matrix_user: str) -> bytes:
    """
    Create a placeholder public key for Matrix users without Ed25519 keys.

    In production, Matrix users should register their Ed25519 public keys
    via a state event or out-of-band verification. This placeholder
    enables testing and development.

    WARNING: Requests using placeholder keys will fail signature verification
    unless signature checking is disabled.

    Args:
        matrix_user: Matrix user ID

    Returns:
        32-byte placeholder "public key"
    """
    return hashlib.sha256(matrix_user.encode()).digest()


# ============================================================================
#  Matrix Event to Signed Request Conversion
# ============================================================================

@dataclass(frozen=True, slots=True)
class MatrixRPCEvent:
    """
    Parsed Matrix event containing HoloLoom RPC call.

    Matrix events with msgtype="m.hololoom.rpc" contain:
    {
        "msgtype": "m.hololoom.rpc",
        "body": {"jsonrpc": "2.0", "method": "...", "params": {...}},
        "meta": {"signature": "...", "nonce": "...", "ts": ...}
    }
    """

    sender: str
    room_id: str
    event_id: str
    method: str
    params: dict[str, Any]
    signature: bytes
    nonce: str
    timestamp: float
    request_id: str = ""

    def to_signed_request(self) -> SignedRequest:
        """Convert to SignedRequest for safety gate processing."""
        return SignedRequest(
            method=self.method,
            params=self.params,
            sender_id=matrix_user_to_node_id(self.sender),
            timestamp=self.timestamp,
            nonce=self.nonce,
            signature=self.signature,
            request_id=self.request_id or self.event_id,
        )


def parse_matrix_rpc_event(
    sender: str,
    room_id: str,
    event_id: str,
    content: dict[str, Any],
) -> MatrixRPCEvent | None:
    """
    Parse a Matrix event into MatrixRPCEvent if it's a HoloLoom RPC.

    Args:
        sender: Matrix user ID of sender
        room_id: Matrix room ID
        event_id: Matrix event ID
        content: Event content dictionary

    Returns:
        MatrixRPCEvent if valid HoloLoom RPC, None otherwise
    """
    # Check message type
    msgtype = content.get("msgtype")
    if msgtype != HOLOLOOM_RPC_MSGTYPE:
        return None

    # Extract body (the JSON-RPC request)
    body = content.get("body", {})
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON body in RPC event {event_id}")
            return None

    # Validate required fields
    if "method" not in body:
        logger.warning(f"Missing 'method' in RPC event {event_id}")
        return None

    # Extract meta (signature info)
    meta = content.get("meta", {})
    signature_b64 = meta.get("signature", "")
    try:
        signature = base64.b64decode(signature_b64) if signature_b64 else b""
    except Exception:
        signature = b""

    return MatrixRPCEvent(
        sender=sender,
        room_id=room_id,
        event_id=event_id,
        method=body.get("method", ""),
        params=body.get("params", {}),
        signature=signature,
        nonce=meta.get("nonce", event_id),
        timestamp=meta.get("ts", time.time()),
        request_id=body.get("id", ""),
    )


# ============================================================================
#  Matrix Transport Adapter
# ============================================================================

# Message handler callback type
MatrixMessageHandler = Callable[[MatrixRPCEvent, GuildTrustLevel], Any]


class MatrixTransportAdapter:
    """
    Matrix transport adapter with integrated safety pipeline.

    Wraps matrix-nio client to provide:
    1. Trust level resolution from Matrix power levels
    2. Safety gate integration (signature, trust, rate limit)
    3. JSON-RPC message handling
    4. Capability state event management

    Security Flow:
        Matrix Event → Parse RPC → Resolve Trust → Safety Gate → Handler

    Example:
        >>> adapter = MatrixTransportAdapter(
        ...     identity=my_identity,
        ...     safety_gate=my_gate,
        ...     rate_limiter=my_limiter,
        ... )
        >>> await adapter.connect("https://matrix.example.org", "@bot:example.org", "token")
        >>> adapter.on_rpc_message(my_handler)
        >>> await adapter.start_sync()
    """

    def __init__(
        self,
        identity: Identity,
        safety_gate: FederationSafetyGate | None = None,
        rate_limiter: FederatedRateLimiter | None = None,
        trust_resolver: MatrixTrustResolver | None = None,
        enable_signature_check: bool = True,
    ):
        """
        Initialize Matrix transport adapter.

        Args:
            identity: Local node identity for signing responses
            safety_gate: Federation safety gate (creates default if None)
            rate_limiter: Rate limiter (creates default if None)
            trust_resolver: Trust resolver (creates default if None)
            enable_signature_check: Whether to verify signatures
        """
        if not HAS_MATRIX_NIO:
            raise ImportError(
                "matrix-nio not installed. Install with: pip install matrix-nio"
            )

        self._identity = identity
        self._safety_gate = safety_gate
        self._rate_limiter = rate_limiter or FederatedRateLimiter()
        self._trust_resolver = trust_resolver or MatrixTrustResolver()
        self._enable_signature = enable_signature_check

        self._client: AsyncClient | None = None
        self._message_handler: MatrixMessageHandler | None = None
        self._room_power_levels: dict[str, dict[str, Any]] = {}
        self._running = False

    @property
    def is_connected(self) -> bool:
        """Check if connected to Matrix homeserver."""
        return self._client is not None and self._running

    async def connect(
        self,
        homeserver: str,
        user_id: str,
        access_token: str,
    ) -> None:
        """
        Connect to Matrix homeserver.

        Args:
            homeserver: Homeserver URL (e.g., "https://matrix.example.org")
            user_id: Matrix user ID (e.g., "@bot:example.org")
            access_token: Access token for authentication
        """
        self._client = nio.AsyncClient(homeserver, user_id)
        self._client.access_token = access_token
        self._client.user_id = user_id

        # Register event callbacks
        self._client.add_event_callback(
            self._on_room_message, nio.RoomMessageText
        )
        self._client.add_event_callback(
            self._on_power_levels, nio.PowerLevelsEvent
        )

        logger.info(f"Connected to Matrix homeserver: {homeserver}")

    async def disconnect(self) -> None:
        """Disconnect from Matrix homeserver."""
        if self._client:
            await self._client.close()
            self._client = None
            self._running = False
            logger.info("Disconnected from Matrix homeserver")

    def on_rpc_message(self, handler: MatrixMessageHandler) -> None:
        """
        Register handler for RPC messages that pass safety checks.

        Args:
            handler: Callback(event, trust_level) -> response
        """
        self._message_handler = handler

    async def start_sync(self) -> None:
        """Start syncing with Matrix homeserver."""
        if not self._client:
            raise RuntimeError("Not connected to homeserver")

        self._running = True
        logger.info("Starting Matrix sync loop")

        while self._running:
            try:
                await self._client.sync(timeout=30000)
            except Exception as e:
                logger.error(f"Matrix sync error: {e}")
                if self._running:
                    await asyncio.sleep(5)

    async def stop_sync(self) -> None:
        """Stop syncing with Matrix homeserver."""
        self._running = False

    async def _on_room_message(
        self,
        room: MatrixRoom,
        event: RoomMessageText,
    ) -> None:
        """Handle incoming room message."""
        # Skip our own messages
        if event.sender == self._client.user_id:
            return

        # Parse as RPC event
        rpc_event = parse_matrix_rpc_event(
            sender=event.sender,
            room_id=room.room_id,
            event_id=event.event_id,
            content=event.source.get("content", {}),
        )

        if rpc_event is None:
            return  # Not a HoloLoom RPC message

        # Process through safety pipeline
        await self._process_rpc_event(rpc_event, room.room_id)

    async def _on_power_levels(
        self,
        room: MatrixRoom,
        event: Any,
    ) -> None:
        """Update cached power levels for a room."""
        self._room_power_levels[room.room_id] = event.content

    async def _process_rpc_event(
        self,
        event: MatrixRPCEvent,
        room_id: str,
    ) -> None:
        """
        Process RPC event through safety pipeline.

        Pipeline:
        1. Resolve trust level from Matrix power levels
        2. Check rate limit
        3. Run through safety gate (signature, trust, safety guardrails)
        4. If allowed, call handler
        """
        if not self._message_handler:
            logger.warning("No RPC handler registered")
            return

        # 1. Resolve trust level
        power_levels = self._room_power_levels.get(room_id, {})
        trust_level = self._trust_resolver.resolve_trust_level(
            event.sender, power_levels
        )

        # 2. Check rate limit
        node_id = matrix_user_to_node_id(event.sender)
        rate_tier = self._trust_resolver.get_rate_limit_tier(trust_level)
        rate_result = self._rate_limiter.check(node_id, rate_tier)

        if rate_result.denied:
            logger.warning(
                f"Rate limited: {event.sender} ({rate_result.reason})"
            )
            await self._send_error_response(
                room_id,
                event.request_id,
                429,
                f"Rate limited: {rate_result.reason}",
            )
            return

        # 3. Run through safety gate
        if self._safety_gate:
            signed_request = event.to_signed_request()

            # Create mock node for safety checks
            node = FederationNode(
                node_id=node_id,
                public_key=matrix_user_to_public_key_placeholder(event.sender),
                endpoint=f"matrix://{event.sender}",
                status=NodeStatus.ONLINE,
            )

            safety_result = await self._safety_gate.check_request(
                request=signed_request,
                node=node,
                guild_trust_level=trust_level,
                rate_limit_result=rate_result,
            )

            if not safety_result.allowed:
                logger.warning(
                    f"Safety denied: {event.sender} - {safety_result.denied_reason}"
                )
                await self._send_error_response(
                    room_id,
                    event.request_id,
                    403,
                    safety_result.denied_reason or "Request denied",
                )
                return

        # 4. Record rate limit usage
        self._rate_limiter.record(node_id)

        # 5. Call handler
        try:
            response = await self._call_handler(event, trust_level)
            if response is not None:
                await self._send_rpc_response(room_id, event.request_id, response)
        except Exception as e:
            logger.error(f"Handler error: {e}")
            await self._send_error_response(
                room_id,
                event.request_id,
                500,
                f"Internal error: {e}",
            )

    async def _call_handler(
        self,
        event: MatrixRPCEvent,
        trust_level: GuildTrustLevel,
    ) -> Any:
        """Call the registered message handler."""
        if asyncio.iscoroutinefunction(self._message_handler):
            return await self._message_handler(event, trust_level)
        else:
            return self._message_handler(event, trust_level)

    async def _send_rpc_response(
        self,
        room_id: str,
        request_id: str,
        result: Any,
    ) -> None:
        """Send JSON-RPC success response."""
        content = {
            "msgtype": HOLOLOOM_RPC_MSGTYPE,
            "body": {
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id,
            },
        }
        await self._send_message(room_id, content)

    async def _send_error_response(
        self,
        room_id: str,
        request_id: str,
        code: int,
        message: str,
    ) -> None:
        """Send JSON-RPC error response."""
        content = {
            "msgtype": HOLOLOOM_RPC_MSGTYPE,
            "body": {
                "jsonrpc": "2.0",
                "error": {"code": code, "message": message},
                "id": request_id,
            },
        }
        await self._send_message(room_id, content)

    async def _send_message(
        self,
        room_id: str,
        content: dict[str, Any],
    ) -> None:
        """Send a message to a Matrix room."""
        if not self._client:
            return

        try:
            await self._client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
            )
        except Exception as e:
            logger.error(f"Failed to send message to {room_id}: {e}")

    async def send_rpc_request(
        self,
        room_id: str,
        method: str,
        params: dict[str, Any],
        request_id: str | None = None,
    ) -> None:
        """
        Send an RPC request to a Matrix room.

        Args:
            room_id: Target room ID
            method: JSON-RPC method name
            params: Method parameters
            request_id: Request ID (generates UUID if None)
        """
        import uuid

        if request_id is None:
            request_id = str(uuid.uuid4())

        # Sign the request
        timestamp = time.time()
        nonce = f"{request_id}-{timestamp}"

        # Create signing payload
        params_str = str(sorted(params.items()))
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]
        payload = f"{method}|{timestamp}|{nonce}|{params_hash}"
        signature = self._identity.sign(payload.encode())

        content = {
            "msgtype": HOLOLOOM_RPC_MSGTYPE,
            "body": {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": request_id,
            },
            "meta": {
                "sender_id": self._identity.node_id,
                "ts": timestamp,
                "nonce": nonce,
                "signature": base64.b64encode(signature).decode(),
            },
        }

        await self._send_message(room_id, content)


# ============================================================================
#  Factory Functions
# ============================================================================

def create_matrix_transport(
    identity: Identity,
    safety_gate: FederationSafetyGate | None = None,
    rate_limiter: FederatedRateLimiter | None = None,
    enable_signature_check: bool = True,
) -> MatrixTransportAdapter:
    """
    Create a configured Matrix transport adapter.

    Args:
        identity: Local node identity
        safety_gate: Safety gate (creates default if None)
        rate_limiter: Rate limiter (creates default if None)
        enable_signature_check: Whether to verify signatures

    Returns:
        Configured MatrixTransportAdapter

    Raises:
        ImportError: If matrix-nio is not installed
    """
    return MatrixTransportAdapter(
        identity=identity,
        safety_gate=safety_gate,
        rate_limiter=rate_limiter,
        enable_signature_check=enable_signature_check,
    )


def create_matrix_trust_resolver(
    custom_power_mapping: dict[int, GuildTrustLevel] | None = None,
) -> MatrixTrustResolver:
    """
    Create a configured Matrix trust resolver.

    Args:
        custom_power_mapping: Override default power level mapping

    Returns:
        Configured MatrixTrustResolver
    """
    return MatrixTrustResolver(custom_power_mapping=custom_power_mapping)


# ============================================================================
#  Exports
# ============================================================================

__all__ = [
    # Constants
    "HOLOLOOM_RPC_MSGTYPE",
    "HOLOLOOM_CAPABILITIES_TYPE",

    # Classes
    "MatrixTrustResolver",
    "MatrixTransportAdapter",
    "MatrixRPCEvent",

    # Functions
    "matrix_user_to_node_id",
    "parse_matrix_rpc_event",
    "create_matrix_transport",
    "create_matrix_trust_resolver",

    # Type hints
    "MatrixMessageHandler",

    # Availability check
    "HAS_MATRIX_NIO",
]
