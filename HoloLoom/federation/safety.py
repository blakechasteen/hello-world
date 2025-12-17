#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federation Safety Layer
========================

Part of Phase 4: Open & Safe Agent Federation (December 2025)
Day 15: Safety Layer Foundation

This module provides comprehensive safety mechanisms for federated agent
communication. Every federated action flows through this safety pipeline:

    Request → Ed25519 Verify → Guild Trust → Rate Limit → SafetyGuardrails → Audit → Execute

Components:
    - SignatureVerifier: Ed25519 cryptographic signature validation
    - GuildTrustChecker: Trust level verification and permission gating
    - FederationSafetyGate: Main orchestrator combining all safety checks

Philosophy:
    "Open protocols, safe by design" - Any agent can speak the protocol,
    but all actions are gated by cryptographic identity, trust levels,
    and safety guardrails.

Author: Claude Code (Phase 4 - December 2025)
"""

from __future__ import annotations

import base64
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, TYPE_CHECKING

from HoloLoom.federation.types import (
    FederationNode,
    GuildTrustLevel,
    Query,
    Response,
    FederationError,
)
from HoloLoom.federation.identity import Identity, SignedMessage

# Import cryptography for direct signature verification
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

if TYPE_CHECKING:
    from HoloLoom.alignment.safety_guardrails import SafetyGuardrails, SafetyDecision
    from HoloLoom.alignment.audit_trail import AuditTrail


# ============================================================================
#  Constants
# ============================================================================

# Nonce expiry to prevent replay attacks (5 minutes)
NONCE_EXPIRY_SECONDS = 300

# Maximum timestamp drift allowed (30 seconds)
MAX_TIMESTAMP_DRIFT_SECONDS = 30


# ============================================================================
#  Trust Level Permissions
# ============================================================================

class FederationPermission(Enum):
    """Permissions for federated operations."""

    # Read-only operations
    RECALL = auto()           # Query knowledge
    CAPABILITIES = auto()     # Query node capabilities

    # Write operations
    GENERATE = auto()         # Generate responses
    STORE = auto()            # Store memories

    # Advanced operations
    UPDATE_PRIORS = auto()    # Update Thompson Sampling priors
    BATCH_OPS = auto()        # Batch operations

    # Administrative operations
    ADMIN = auto()            # Administrative actions
    ROUTING = auto()          # Routing decisions


# Trust level to permissions mapping
TRUST_PERMISSIONS: Dict[GuildTrustLevel, FrozenSet[FederationPermission]] = {
    GuildTrustLevel.STARTER: frozenset({
        FederationPermission.RECALL,
        FederationPermission.CAPABILITIES,
    }),
    GuildTrustLevel.ESTABLISHED: frozenset({
        FederationPermission.RECALL,
        FederationPermission.CAPABILITIES,
        FederationPermission.GENERATE,
        FederationPermission.STORE,
    }),
    GuildTrustLevel.VETERAN: frozenset({
        FederationPermission.RECALL,
        FederationPermission.CAPABILITIES,
        FederationPermission.GENERATE,
        FederationPermission.STORE,
        FederationPermission.UPDATE_PRIORS,
        FederationPermission.BATCH_OPS,
        FederationPermission.ADMIN,
        FederationPermission.ROUTING,
    }),
}

# Method to permission mapping
METHOD_PERMISSIONS: Dict[str, FederationPermission] = {
    # RAG methods
    "hololoom.rag.recall": FederationPermission.RECALL,
    "hololoom.rag.store": FederationPermission.STORE,

    # Inference methods
    "hololoom.inference.generate": FederationPermission.GENERATE,
    "hololoom.inference.capabilities": FederationPermission.CAPABILITIES,

    # Pattern methods
    "hololoom.pattern.select": FederationPermission.RECALL,
    "hololoom.pattern.complexity": FederationPermission.RECALL,

    # Thread methods
    "hololoom.threads.select": FederationPermission.RECALL,
    "hololoom.threads.expand": FederationPermission.RECALL,

    # Feature methods
    "hololoom.features.extract": FederationPermission.GENERATE,
    "hololoom.features.dim": FederationPermission.CAPABILITIES,

    # Convergence methods
    "hololoom.convergence.collapse": FederationPermission.GENERATE,
    "hololoom.convergence.update": FederationPermission.UPDATE_PRIORS,

    # Tool methods
    "hololoom.tool.execute": FederationPermission.GENERATE,
    "hololoom.tool.list": FederationPermission.CAPABILITIES,

    # Spacetime methods
    "hololoom.spacetime.assemble": FederationPermission.GENERATE,

    # Admin methods
    "hololoom.admin.nodes": FederationPermission.ADMIN,
    "hololoom.admin.routing": FederationPermission.ROUTING,
}


# ============================================================================
#  Data Classes
# ============================================================================

@dataclass(frozen=True, slots=True)
class SignedRequest:
    """
    A federation request with cryptographic signature.

    Format follows JSON-RPC 2.0 with signed metadata:
    {
        "jsonrpc": "2.0",
        "method": "hololoom.rag.recall",
        "params": {...},
        "meta": {
            "sender_id": "ed25519:abc123...",
            "timestamp": 1702300800,
            "nonce": "unique-request-id",
            "signature": "base64-encoded-sig"
        },
        "id": "req-001"
    }
    """

    method: str
    params: Dict[str, Any]
    sender_id: str
    timestamp: float
    nonce: str
    signature: bytes
    request_id: str = ""

    def get_signing_payload(self) -> bytes:
        """Get the payload that was signed."""
        # Canonical format: method|timestamp|nonce|params_hash
        params_str = str(sorted(self.params.items()))
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]
        payload = f"{self.method}|{self.timestamp}|{self.nonce}|{params_hash}"
        return payload.encode()


@dataclass(frozen=True, slots=True)
class SafetyCheckResult:
    """Result of a federation safety check."""

    allowed: bool
    reason: str
    check_type: str  # "signature", "trust", "rate_limit", "safety"
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def denied(self) -> bool:
        return not self.allowed


@dataclass(slots=True)
class FederationSafetyResult:
    """Complete result of all federation safety checks."""

    allowed: bool
    checks: List[SafetyCheckResult]
    sender_id: str
    method: str
    request_id: str
    total_check_time_ms: float = 0.0

    @property
    def denied_reason(self) -> Optional[str]:
        """Get the reason for denial if denied."""
        for check in self.checks:
            if check.denied:
                return f"{check.check_type}: {check.reason}"
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "allowed": self.allowed,
            "sender_id": self.sender_id,
            "method": self.method,
            "request_id": self.request_id,
            "total_check_time_ms": self.total_check_time_ms,
            "checks": [
                {
                    "type": c.check_type,
                    "allowed": c.allowed,
                    "reason": c.reason,
                }
                for c in self.checks
            ],
            "denied_reason": self.denied_reason,
        }


# ============================================================================
#  Signature Verifier
# ============================================================================

class SignatureVerifier:
    """
    Ed25519 signature verification for federation requests.

    Verifies that requests are properly signed by the claimed sender
    and prevents replay attacks through nonce tracking.

    Example:
        >>> verifier = SignatureVerifier()
        >>> result = verifier.verify(signed_request, sender_public_key)
        >>> if result.allowed:
        ...     # Proceed with request
    """

    def __init__(
        self,
        nonce_expiry_seconds: float = NONCE_EXPIRY_SECONDS,
        max_timestamp_drift_seconds: float = MAX_TIMESTAMP_DRIFT_SECONDS,
    ):
        self._nonce_expiry = nonce_expiry_seconds
        self._max_drift = max_timestamp_drift_seconds
        self._seen_nonces: Dict[str, float] = {}  # nonce -> timestamp
        self._last_cleanup = time.time()

    def verify(
        self,
        request: SignedRequest,
        sender_public_key: bytes,
    ) -> SafetyCheckResult:
        """
        Verify a signed federation request.

        Checks:
        1. Timestamp is within acceptable drift
        2. Nonce has not been seen (replay prevention)
        3. Signature is valid for the payload

        Args:
            request: The signed request to verify
            sender_public_key: Ed25519 public key of claimed sender

        Returns:
            SafetyCheckResult with verification outcome
        """
        # Cleanup old nonces periodically
        self._cleanup_nonces()

        # Check timestamp drift
        now = time.time()
        drift = abs(now - request.timestamp)
        if drift > self._max_drift:
            return SafetyCheckResult(
                allowed=False,
                reason=f"Timestamp drift too large: {drift:.1f}s (max: {self._max_drift}s)",
                check_type="signature",
                details={"drift_seconds": drift},
            )

        # Check for replay (nonce reuse)
        if request.nonce in self._seen_nonces:
            return SafetyCheckResult(
                allowed=False,
                reason="Nonce already used (possible replay attack)",
                check_type="signature",
                details={"nonce": request.nonce},
            )

        # Verify signature
        try:
            if not HAS_CRYPTOGRAPHY:
                return SafetyCheckResult(
                    allowed=False,
                    reason="Cryptography library not available for signature verification",
                    check_type="signature",
                )

            payload = request.get_signing_payload()

            # Verify Ed25519 signature directly
            try:
                pub_key = Ed25519PublicKey.from_public_bytes(sender_public_key)
                pub_key.verify(request.signature, payload)
            except Exception:
                return SafetyCheckResult(
                    allowed=False,
                    reason="Invalid signature",
                    check_type="signature",
                )
        except Exception as e:
            return SafetyCheckResult(
                allowed=False,
                reason=f"Signature verification failed: {e}",
                check_type="signature",
            )

        # Record nonce to prevent replay
        self._seen_nonces[request.nonce] = request.timestamp

        return SafetyCheckResult(
            allowed=True,
            reason="Signature valid",
            check_type="signature",
            details={"sender_id": request.sender_id},
        )

    def _cleanup_nonces(self) -> None:
        """Remove expired nonces."""
        now = time.time()
        if now - self._last_cleanup < 60:  # Cleanup every minute
            return

        cutoff = now - self._nonce_expiry
        self._seen_nonces = {
            nonce: ts
            for nonce, ts in self._seen_nonces.items()
            if ts > cutoff
        }
        self._last_cleanup = now


# ============================================================================
#  Guild Trust Checker
# ============================================================================

class GuildTrustChecker:
    """
    Verifies guild trust levels and permissions for federation requests.

    Maps trust levels to allowed operations:
    - STARTER: recall, capabilities only
    - ESTABLISHED: + generate, store
    - VETERAN: + update_priors, batch_ops, admin, routing

    Example:
        >>> checker = GuildTrustChecker()
        >>> result = checker.check(node, "hololoom.rag.recall")
        >>> if result.allowed:
        ...     # Node has permission
    """

    def __init__(
        self,
        custom_permissions: Optional[Dict[GuildTrustLevel, FrozenSet[FederationPermission]]] = None,
    ):
        self._permissions = custom_permissions or TRUST_PERMISSIONS

    def check(
        self,
        node: FederationNode,
        method: str,
        guild_trust_level: Optional[GuildTrustLevel] = None,
    ) -> SafetyCheckResult:
        """
        Check if a node has permission to call a method.

        Args:
            node: The federation node making the request
            method: The JSON-RPC method being called
            guild_trust_level: Override trust level (uses node's guilds if None)

        Returns:
            SafetyCheckResult with permission check outcome
        """
        # Get required permission for method
        required_permission = METHOD_PERMISSIONS.get(method)
        if required_permission is None:
            return SafetyCheckResult(
                allowed=False,
                reason=f"Unknown method: {method}",
                check_type="trust",
                details={"method": method},
            )

        # Determine trust level
        trust_level = guild_trust_level
        if trust_level is None:
            # Use highest trust level from node's guilds
            # For now, default to STARTER if no guild info
            trust_level = GuildTrustLevel.STARTER

        # Check permission
        allowed_permissions = self._permissions.get(trust_level, frozenset())

        if required_permission in allowed_permissions:
            return SafetyCheckResult(
                allowed=True,
                reason=f"Permission granted for {trust_level.name}",
                check_type="trust",
                details={
                    "trust_level": trust_level.name,
                    "permission": required_permission.name,
                },
            )

        return SafetyCheckResult(
            allowed=False,
            reason=f"Insufficient trust level: {trust_level.name} lacks {required_permission.name}",
            check_type="trust",
            details={
                "trust_level": trust_level.name,
                "required_permission": required_permission.name,
                "allowed_permissions": [p.name for p in allowed_permissions],
            },
        )

    def get_allowed_methods(self, trust_level: GuildTrustLevel) -> List[str]:
        """Get all methods allowed for a trust level."""
        allowed_permissions = self._permissions.get(trust_level, frozenset())
        return [
            method
            for method, perm in METHOD_PERMISSIONS.items()
            if perm in allowed_permissions
        ]


# ============================================================================
#  Federation Safety Gate
# ============================================================================

class FederationSafetyGate:
    """
    Main safety orchestrator for federation requests.

    Combines all safety checks into a single pipeline:
    1. Signature verification (Ed25519)
    2. Guild trust level check
    3. Rate limiting (via external rate limiter)
    4. Safety guardrails (via alignment framework)
    5. Audit logging

    Example:
        >>> gate = FederationSafetyGate(
        ...     guardrails=safety_guardrails,
        ...     audit_trail=audit_trail,
        ... )
        >>> result = await gate.check_request(signed_request, node)
        >>> if result.allowed:
        ...     # Process request
        ... else:
        ...     print(f"Denied: {result.denied_reason}")
    """

    def __init__(
        self,
        guardrails: Optional['SafetyGuardrails'] = None,
        audit_trail: Optional['AuditTrail'] = None,
        signature_verifier: Optional[SignatureVerifier] = None,
        trust_checker: Optional[GuildTrustChecker] = None,
        enable_signature_check: bool = True,
        enable_trust_check: bool = True,
        enable_safety_check: bool = True,
        enable_audit: bool = True,
    ):
        self._guardrails = guardrails
        self._audit_trail = audit_trail
        self._signature_verifier = signature_verifier or SignatureVerifier()
        self._trust_checker = trust_checker or GuildTrustChecker()

        self._enable_signature = enable_signature_check
        self._enable_trust = enable_trust_check
        self._enable_safety = enable_safety_check
        self._enable_audit = enable_audit

    async def check_request(
        self,
        request: SignedRequest,
        node: FederationNode,
        guild_trust_level: Optional[GuildTrustLevel] = None,
        rate_limit_result: Optional[SafetyCheckResult] = None,
    ) -> FederationSafetyResult:
        """
        Run all safety checks on a federation request.

        Args:
            request: The signed request to check
            node: The federation node making the request
            guild_trust_level: Override trust level
            rate_limit_result: Pre-computed rate limit check result

        Returns:
            FederationSafetyResult with all check outcomes
        """
        start_time = time.time()
        checks: List[SafetyCheckResult] = []
        allowed = True

        # 1. Signature verification
        if self._enable_signature:
            sig_result = self._signature_verifier.verify(request, node.public_key)
            checks.append(sig_result)
            if sig_result.denied:
                allowed = False

        # 2. Guild trust check
        if allowed and self._enable_trust:
            trust_result = self._trust_checker.check(node, request.method, guild_trust_level)
            checks.append(trust_result)
            if trust_result.denied:
                allowed = False

        # 3. Rate limit check (external)
        if rate_limit_result is not None:
            checks.append(rate_limit_result)
            if rate_limit_result.denied:
                allowed = False

        # 4. Safety guardrails
        if allowed and self._enable_safety and self._guardrails is not None:
            safety_result = await self._check_safety(request)
            checks.append(safety_result)
            if safety_result.denied:
                allowed = False

        # Calculate total time
        total_time_ms = (time.time() - start_time) * 1000

        result = FederationSafetyResult(
            allowed=allowed,
            checks=checks,
            sender_id=request.sender_id,
            method=request.method,
            request_id=request.request_id,
            total_check_time_ms=total_time_ms,
        )

        # 5. Audit logging
        if self._enable_audit and self._audit_trail is not None:
            await self._log_audit(result, request, node)

        return result

    async def _check_safety(self, request: SignedRequest) -> SafetyCheckResult:
        """Run safety guardrails on request."""
        try:
            # Import here to avoid circular imports
            from HoloLoom.alignment.safety_guardrails import ActionRequest, ActionCategory

            # Map method to action category
            category = self._method_to_category(request.method)

            # Create action request
            action_request = ActionRequest(
                action=request.method,
                category=category,
                context=request.params,
                requester_id=request.sender_id,
            )

            # Evaluate safety
            decision = self._guardrails.evaluate(action_request)

            if decision.allowed:
                return SafetyCheckResult(
                    allowed=True,
                    reason="Safety check passed",
                    check_type="safety",
                    details={"risk_level": decision.risk_level.value},
                )
            else:
                return SafetyCheckResult(
                    allowed=False,
                    reason=decision.reason or "Safety check failed",
                    check_type="safety",
                    details={
                        "risk_level": decision.risk_level.value,
                        "factors": decision.risk_factors,
                    },
                )
        except Exception as e:
            return SafetyCheckResult(
                allowed=False,
                reason=f"Safety check error: {e}",
                check_type="safety",
            )

    def _method_to_category(self, method: str) -> 'ActionCategory':
        """Map JSON-RPC method to action category."""
        from HoloLoom.alignment.safety_guardrails import ActionCategory

        # Map method prefixes to categories
        if method.startswith("hololoom.rag."):
            return ActionCategory.DATA_ACCESS
        elif method.startswith("hololoom.inference."):
            return ActionCategory.GENERATION
        elif method.startswith("hololoom.admin."):
            return ActionCategory.SYSTEM_OPERATION
        elif method.startswith("hololoom.tool."):
            return ActionCategory.TOOL_USE
        else:
            return ActionCategory.UNKNOWN

    async def _log_audit(
        self,
        result: FederationSafetyResult,
        request: SignedRequest,
        node: FederationNode,
    ) -> None:
        """Log safety decision to audit trail."""
        try:
            self._audit_trail.log_decision(
                decision_type="federation_safety",
                outcome="allowed" if result.allowed else "denied",
                reason=result.denied_reason or "All checks passed",
                context={
                    "sender_id": request.sender_id,
                    "method": request.method,
                    "request_id": request.request_id,
                    "node_id": node.node_id,
                    "checks": [
                        {"type": c.check_type, "allowed": c.allowed}
                        for c in result.checks
                    ],
                    "total_check_time_ms": result.total_check_time_ms,
                },
            )
        except Exception:
            pass  # Don't fail on audit errors


# ============================================================================
#  Factory Functions
# ============================================================================

def create_federation_safety_gate(
    guardrails: Optional['SafetyGuardrails'] = None,
    audit_trail: Optional['AuditTrail'] = None,
    **kwargs,
) -> FederationSafetyGate:
    """
    Create a FederationSafetyGate with optional components.

    Args:
        guardrails: SafetyGuardrails instance (creates default if None)
        audit_trail: AuditTrail instance (creates default if None)
        **kwargs: Additional arguments passed to FederationSafetyGate

    Returns:
        Configured FederationSafetyGate
    """
    # Lazy import to avoid circular dependencies
    if guardrails is None:
        try:
            from HoloLoom.alignment.safety_guardrails import SafetyGuardrails
            guardrails = SafetyGuardrails()
        except ImportError:
            pass

    if audit_trail is None:
        try:
            from HoloLoom.alignment.audit_trail import AuditTrail
            audit_trail = AuditTrail()
        except ImportError:
            pass

    return FederationSafetyGate(
        guardrails=guardrails,
        audit_trail=audit_trail,
        **kwargs,
    )


def parse_signed_request(data: Dict[str, Any]) -> SignedRequest:
    """
    Parse a JSON-RPC request with signed metadata into SignedRequest.

    Args:
        data: Dictionary with jsonrpc, method, params, meta, id

    Returns:
        SignedRequest object

    Raises:
        ValueError: If required fields are missing
    """
    if "method" not in data:
        raise ValueError("Missing 'method' field")

    meta = data.get("meta", {})
    if not meta:
        raise ValueError("Missing 'meta' field with signature data")

    # Decode base64 signature
    sig_b64 = meta.get("signature", "")
    try:
        signature = base64.b64decode(sig_b64)
    except Exception:
        raise ValueError("Invalid base64 signature")

    return SignedRequest(
        method=data["method"],
        params=data.get("params", {}),
        sender_id=meta.get("sender_id", ""),
        timestamp=meta.get("timestamp", 0),
        nonce=meta.get("nonce", ""),
        signature=signature,
        request_id=data.get("id", ""),
    )


# ============================================================================
#  Exports
# ============================================================================

__all__ = [
    # Enums
    "FederationPermission",

    # Data classes
    "SignedRequest",
    "SafetyCheckResult",
    "FederationSafetyResult",

    # Classes
    "SignatureVerifier",
    "GuildTrustChecker",
    "FederationSafetyGate",

    # Factory functions
    "create_federation_safety_gate",
    "parse_signed_request",

    # Constants
    "TRUST_PERMISSIONS",
    "METHOD_PERMISSIONS",
]
