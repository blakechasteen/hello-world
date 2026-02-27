"""
Hardened Handoff Orchestrator - 7-layer security integration.

Defense in depth for cross-device handoff operations:
1. Request Signing (HMAC + nonce + timestamp)
2. Rate Limiting (per-device)
3. Circuit Breakers (device health isolation)
4. WAF Validation (payload sanitization)
5. Risk Gating (SAFE → CRITICAL assessment)
6. Security Monitoring (real-time + alerting)
7. Audit Trail (complete provenance)

Created: 2025-12-08
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .identity import UnifiedIdentity

from .synced_memory import SyncedMemory, MergeResult
from .types import (
    DeviceStatus,
    HandoffRequest,
    HandoffResult,
    HandoffStatus,
    SignedOp,
    HandoffError,
    DeviceNotFoundError,
    DeviceRevokedError,
    DeviceUnhealthyError,
    RateLimitExceededError,
    InvalidSignatureError,
    ReplayAttackError,
    PayloadRejectedError,
    HandoffBlockedError,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  RISK LEVELS
# ═══════════════════════════════════════════════════════════════════════════


class RiskLevel(Enum):
    """Risk levels for handoff operations."""

    SAFE = auto()       # No human review needed
    LOW = auto()        # Minor review
    MEDIUM = auto()     # Standard review
    HIGH = auto()       # Careful review
    CRITICAL = auto()   # Block without explicit approval


# ═══════════════════════════════════════════════════════════════════════════
#  SECURITY LAYER PROTOCOLS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RateLimitResult:
    """Result of rate limit check."""

    allowed: bool
    remaining: int = 0
    reset_at: float = 0.0
    device_id: str = ""


@dataclass
class WAFResult:
    """Result of WAF validation."""

    passed: bool
    rule_id: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class RiskAssessment:
    """Result of risk assessment."""

    level: RiskLevel
    reason: str
    should_block: bool = False
    requires_approval: bool = False


# ═══════════════════════════════════════════════════════════════════════════
#  SIMPLE IN-MEMORY RATE LIMITER (Fallback)
# ═══════════════════════════════════════════════════════════════════════════


class InMemoryRateLimiter:
    """
    Simple token bucket rate limiter for per-device limits.

    Used as fallback when Redis is not available.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
    ):
        self.rpm = requests_per_minute
        self.burst = burst_size
        self._buckets: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def check(self, device_key: str) -> RateLimitResult:
        """Check if request is allowed under rate limit."""
        async with self._lock:
            now = time.time()

            if device_key not in self._buckets:
                self._buckets[device_key] = {
                    "tokens": self.burst,
                    "last_update": now,
                }

            bucket = self._buckets[device_key]

            # Refill tokens based on time elapsed
            elapsed = now - bucket["last_update"]
            refill = (elapsed / 60.0) * self.rpm
            bucket["tokens"] = min(self.burst, bucket["tokens"] + refill)
            bucket["last_update"] = now

            if bucket["tokens"] >= 1.0:
                bucket["tokens"] -= 1.0
                return RateLimitResult(
                    allowed=True,
                    remaining=int(bucket["tokens"]),
                    reset_at=now + 60.0,
                    device_id=device_key,
                )
            else:
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=now + (60.0 / self.rpm),
                    device_id=device_key,
                )


# ═══════════════════════════════════════════════════════════════════════════
#  SIMPLE CIRCUIT BREAKER (Per-Device)
# ═══════════════════════════════════════════════════════════════════════════


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, blocking requests
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class DeviceCircuit:
    """Per-device circuit breaker state."""

    device_id: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure: float = 0.0
    last_transition: float = field(default_factory=time.time)


class DeviceCircuitBreaker:
    """
    Per-device circuit breaker for isolating unhealthy devices.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self._circuits: Dict[str, DeviceCircuit] = {}
        self._lock = asyncio.Lock()

    async def get_circuit(self, device_id: str) -> DeviceCircuit:
        """Get or create circuit for device."""
        async with self._lock:
            if device_id not in self._circuits:
                self._circuits[device_id] = DeviceCircuit(device_id=device_id)
            return self._circuits[device_id]

    async def is_open(self, device_id: str) -> bool:
        """Check if circuit is open (blocking requests)."""
        circuit = await self.get_circuit(device_id)

        if circuit.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - circuit.last_failure >= self.recovery_timeout:
                circuit.state = CircuitState.HALF_OPEN
                circuit.success_count = 0
                circuit.last_transition = time.time()
                return False
            return True

        return False

    async def record_success(self, device_id: str) -> None:
        """Record successful operation."""
        circuit = await self.get_circuit(device_id)

        if circuit.state == CircuitState.HALF_OPEN:
            circuit.success_count += 1
            if circuit.success_count >= self.success_threshold:
                circuit.state = CircuitState.CLOSED
                circuit.failure_count = 0
                circuit.last_transition = time.time()
        elif circuit.state == CircuitState.CLOSED:
            circuit.failure_count = 0

    async def record_failure(self, device_id: str) -> None:
        """Record failed operation."""
        circuit = await self.get_circuit(device_id)
        circuit.failure_count += 1
        circuit.last_failure = time.time()

        if circuit.state == CircuitState.HALF_OPEN:
            circuit.state = CircuitState.OPEN
            circuit.last_transition = time.time()
        elif circuit.state == CircuitState.CLOSED:
            if circuit.failure_count >= self.failure_threshold:
                circuit.state = CircuitState.OPEN
                circuit.last_transition = time.time()


# ═══════════════════════════════════════════════════════════════════════════
#  SIMPLE WAF (Payload Validation)
# ═══════════════════════════════════════════════════════════════════════════


class SimpleWAF:
    """
    Simple Web Application Firewall for payload validation.

    Checks for common attack patterns in handoff payloads.
    """

    # Patterns that indicate malicious payloads
    BLOCKED_PATTERNS = [
        # SQL injection
        ("sql_injection", ["'; DROP", "' OR '1'='1", "UNION SELECT"]),
        # Path traversal
        ("path_traversal", ["../", "..\\", "%2e%2e"]),
        # Script injection
        ("script_injection", ["<script", "javascript:", "onerror="]),
        # Command injection
        ("command_injection", ["; rm -rf", "| cat /etc", "&& wget"]),
    ]

    def validate(self, payload: Dict[str, Any]) -> WAFResult:
        """Validate payload for malicious content."""
        payload_str = str(payload).lower()

        for rule_id, patterns in self.BLOCKED_PATTERNS:
            for pattern in patterns:
                if pattern.lower() in payload_str:
                    return WAFResult(
                        passed=False,
                        rule_id=rule_id,
                        reason=f"Blocked by WAF rule: {rule_id}",
                    )

        return WAFResult(passed=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SIMPLE RISK ASSESSOR
# ═══════════════════════════════════════════════════════════════════════════


class SimpleRiskAssessor:
    """
    Simple risk assessment for handoff operations.

    Assesses risk based on:
    - Context sensitivity
    - Operation type
    - Device history
    """

    def assess(
        self,
        context: Dict[str, Any],
        device_history: Optional[Dict[str, Any]] = None,
    ) -> RiskAssessment:
        """Assess risk level of handoff operation."""

        # Check for sensitive content markers
        sensitive_keywords = ["password", "secret", "key", "token", "credential"]
        context_str = str(context).lower()

        has_sensitive = any(kw in context_str for kw in sensitive_keywords)

        # Check operation count
        ops_count = len(context.get("pending_ops", []))

        # Assess risk
        if has_sensitive:
            return RiskAssessment(
                level=RiskLevel.HIGH,
                reason="Contains sensitive content",
                requires_approval=True,
            )
        elif ops_count > 100:
            return RiskAssessment(
                level=RiskLevel.MEDIUM,
                reason=f"Large operation batch ({ops_count} ops)",
            )
        elif ops_count > 1000:
            return RiskAssessment(
                level=RiskLevel.CRITICAL,
                reason=f"Very large batch ({ops_count} ops) - potential abuse",
                should_block=True,
            )
        else:
            return RiskAssessment(
                level=RiskLevel.SAFE,
                reason="Normal operation",
            )


# ═══════════════════════════════════════════════════════════════════════════
#  SECURITY EVENT LOGGER
# ═══════════════════════════════════════════════════════════════════════════


class SecurityEventType(Enum):
    """Types of security events."""

    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    RATE_LIMIT = "rate_limit"
    CIRCUIT_OPEN = "circuit_open"
    WAF_BLOCK = "waf_block"
    RISK_HIGH = "risk_high"
    HANDOFF_COMPLETE = "handoff_complete"
    HANDOFF_FAILED = "handoff_failed"


@dataclass
class SecurityEvent:
    """A security event for monitoring."""

    event_type: SecurityEventType
    timestamp: float
    device_id: str
    identity_did: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "device_id": self.device_id,
            "identity_did": self.identity_did,
            "details": self.details,
        }


class SecurityMonitor:
    """
    Simple security event monitor.

    Logs security events and provides alerting hooks.
    """

    def __init__(self, alert_callback: Optional[Callable[[SecurityEvent], None]] = None):
        self.events: List[SecurityEvent] = []
        self.alert_callback = alert_callback
        self._max_events = 10000

    def log_event(self, event: SecurityEvent) -> None:
        """Log a security event."""
        self.events.append(event)

        # Trim old events
        if len(self.events) > self._max_events:
            self.events = self.events[-self._max_events:]

        # Alert on high-severity events
        if event.event_type in (
            SecurityEventType.AUTH_FAILURE,
            SecurityEventType.WAF_BLOCK,
            SecurityEventType.RISK_HIGH,
        ):
            if self.alert_callback:
                try:
                    self.alert_callback(event)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

        # Log to standard logger
        logger.info(f"Security event: {event.event_type.value} - {event.details}")

    def get_recent_events(
        self,
        event_type: Optional[SecurityEventType] = None,
        limit: int = 100,
    ) -> List[SecurityEvent]:
        """Get recent security events."""
        events = self.events
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]


# ═══════════════════════════════════════════════════════════════════════════
#  AUDIT TRAIL
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class HandoffAuditEntry:
    """Audit trail entry for handoff operations."""

    entry_id: str
    timestamp: float
    source_device: str
    target_device: str
    identity_did: str
    outcome: str
    ops_count: int
    duration_ms: float
    risk_level: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "source_device": self.source_device,
            "target_device": self.target_device,
            "identity_did": self.identity_did,
            "outcome": self.outcome,
            "ops_count": self.ops_count,
            "duration_ms": self.duration_ms,
            "risk_level": self.risk_level,
            "details": self.details,
        }


class HandoffAuditTrail:
    """
    Audit trail for handoff operations.

    Provides complete provenance of all handoff decisions.
    """

    def __init__(self):
        self.entries: List[HandoffAuditEntry] = []
        self._max_entries = 10000

    def log(
        self,
        source_device: str,
        target_device: str,
        identity_did: str,
        outcome: str,
        ops_count: int,
        duration_ms: float,
        risk_level: RiskLevel,
        details: Optional[Dict[str, Any]] = None,
    ) -> HandoffAuditEntry:
        """Log a handoff operation."""
        entry = HandoffAuditEntry(
            entry_id=f"audit_{secrets.token_hex(8)}",
            timestamp=time.time(),
            source_device=source_device,
            target_device=target_device,
            identity_did=identity_did,
            outcome=outcome,
            ops_count=ops_count,
            duration_ms=duration_ms,
            risk_level=risk_level.name,
            details=details or {},
        )
        self.entries.append(entry)

        # Trim old entries
        if len(self.entries) > self._max_entries:
            self.entries = self.entries[-self._max_entries:]

        return entry

    def query(
        self,
        identity_did: Optional[str] = None,
        device_id: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[HandoffAuditEntry]:
        """Query audit trail."""
        results = self.entries

        if identity_did:
            results = [e for e in results if e.identity_did == identity_did]
        if device_id:
            results = [
                e for e in results
                if e.source_device == device_id or e.target_device == device_id
            ]
        if since:
            results = [e for e in results if e.timestamp >= since]

        return results[-limit:]


# ═══════════════════════════════════════════════════════════════════════════
#  HARDENED HANDOFF ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════


class HardenedHandoffOrchestrator:
    """
    Secure cross-device handoff with 7-layer defense in depth.

    Integrates:
    1. Request Signing - Ed25519 signatures
    2. Rate Limiting - Per-device throttling
    3. Circuit Breakers - Unhealthy device isolation
    4. WAF Validation - Payload sanitization
    5. Risk Gating - SAFE → CRITICAL assessment
    6. Security Monitoring - Real-time event logging
    7. Audit Trail - Complete provenance

    Usage:
        identity = UnifiedIdentity.create("blake", "laptop")
        orchestrator = HardenedHandoffOrchestrator(identity)

        # Handoff to another device
        result = await orchestrator.handoff_to("phone_device_id", context)

        # Receive handoff from another device
        await orchestrator.receive_handoff(signed_request)
    """

    def __init__(
        self,
        identity: "UnifiedIdentity",
        memory: Optional[SyncedMemory] = None,
        rate_limiter: Optional[InMemoryRateLimiter] = None,
        circuit_breaker: Optional[DeviceCircuitBreaker] = None,
        waf: Optional[SimpleWAF] = None,
        risk_assessor: Optional[SimpleRiskAssessor] = None,
        security_monitor: Optional[SecurityMonitor] = None,
        audit_trail: Optional[HandoffAuditTrail] = None,
    ):
        self.identity = identity
        self.memory = memory or SyncedMemory(identity)

        # Security layers (with defaults)
        self.rate_limiter = rate_limiter or InMemoryRateLimiter()
        self.circuit_breaker = circuit_breaker or DeviceCircuitBreaker()
        self.waf = waf or SimpleWAF()
        self.risk_assessor = risk_assessor or SimpleRiskAssessor()
        self.monitor = security_monitor or SecurityMonitor()
        self.audit = audit_trail or HandoffAuditTrail()

    async def handoff_to(
        self,
        target_device: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> HandoffResult:
        """
        Execute handoff to another device with 7-layer security.

        Args:
            target_device: Device ID of target
            context: Optional context to transfer

        Returns:
            HandoffResult with status and statistics

        Raises:
            RateLimitExceededError: Rate limit exceeded
            DeviceUnhealthyError: Circuit breaker open
            PayloadRejectedError: WAF blocked payload
            HandoffBlockedError: Risk too high
            DeviceNotFoundError: Target device not found
            DeviceRevokedError: Target device revoked
        """
        start_time = time.time()
        context = context or {}
        source_device = self.identity.current_device_id or "unknown"

        try:
            # ─────────────────────────────────────────────────────────────
            # Layer 1: Verify target device exists and is authorized
            # ─────────────────────────────────────────────────────────────
            if not self.identity.is_device_authorized(target_device):
                device = self.identity.devices.get(target_device)
                if device is None:
                    raise DeviceNotFoundError(
                        "Target device not found",
                        device_id=target_device,
                        suggestion="Pair device first with add_device()",
                    )
                elif device.status == DeviceStatus.REVOKED:
                    raise DeviceRevokedError(
                        "Target device has been revoked",
                        device_id=target_device,
                    )
                else:
                    raise DeviceUnhealthyError(
                        f"Target device status: {device.status.name}",
                        device_id=target_device,
                    )

            # ─────────────────────────────────────────────────────────────
            # Layer 2: Rate limit check
            # ─────────────────────────────────────────────────────────────
            rate_key = f"handoff:{self.identity.did}:{target_device}"
            rate_result = await self.rate_limiter.check(rate_key)

            if not rate_result.allowed:
                self.monitor.log_event(SecurityEvent(
                    event_type=SecurityEventType.RATE_LIMIT,
                    timestamp=time.time(),
                    device_id=target_device,
                    identity_did=self.identity.did,
                    details={"remaining": rate_result.remaining},
                ))
                raise RateLimitExceededError(
                    f"Rate limit exceeded for device {target_device}",
                    device_id=target_device,
                    suggestion=f"Wait until {rate_result.reset_at}",
                )

            # ─────────────────────────────────────────────────────────────
            # Layer 3: Circuit breaker check
            # ─────────────────────────────────────────────────────────────
            if await self.circuit_breaker.is_open(target_device):
                self.monitor.log_event(SecurityEvent(
                    event_type=SecurityEventType.CIRCUIT_OPEN,
                    timestamp=time.time(),
                    device_id=target_device,
                    identity_did=self.identity.did,
                ))
                raise DeviceUnhealthyError(
                    f"Circuit breaker open for device {target_device}",
                    device_id=target_device,
                    suggestion="Wait for recovery timeout",
                )

            # ─────────────────────────────────────────────────────────────
            # Layer 4: WAF validation
            # ─────────────────────────────────────────────────────────────
            waf_result = self.waf.validate(context)
            if not waf_result.passed:
                self.monitor.log_event(SecurityEvent(
                    event_type=SecurityEventType.WAF_BLOCK,
                    timestamp=time.time(),
                    device_id=target_device,
                    identity_did=self.identity.did,
                    details={"rule_id": waf_result.rule_id},
                ))
                raise PayloadRejectedError(
                    waf_result.reason or "WAF rejected payload",
                    device_id=target_device,
                )

            # ─────────────────────────────────────────────────────────────
            # Layer 5: Risk assessment
            # ─────────────────────────────────────────────────────────────
            pending_ops = self.memory.pending_delta()
            risk_context = {
                "pending_ops": [op.to_dict() for op in pending_ops],
                **context,
            }
            risk = self.risk_assessor.assess(risk_context)

            if risk.should_block:
                self.monitor.log_event(SecurityEvent(
                    event_type=SecurityEventType.RISK_HIGH,
                    timestamp=time.time(),
                    device_id=target_device,
                    identity_did=self.identity.did,
                    details={"risk_level": risk.level.name, "reason": risk.reason},
                ))
                raise HandoffBlockedError(
                    f"Handoff blocked due to {risk.level.name} risk: {risk.reason}",
                    device_id=target_device,
                )

            if risk.level == RiskLevel.CRITICAL:
                raise HandoffBlockedError(
                    f"Critical risk level: {risk.reason}",
                    device_id=target_device,
                )

            # ─────────────────────────────────────────────────────────────
            # Create and sign handoff request
            # ─────────────────────────────────────────────────────────────
            request = HandoffRequest(
                request_id="",  # Will be auto-generated
                source_device=source_device,
                target_device=target_device,
                identity_did=self.identity.did,
                pending_ops=[op.to_dict() for op in pending_ops],
                context=context,
                timestamp=0.0,  # Will be auto-generated
                nonce="",  # Will be auto-generated
            )

            # Sign the request
            signed_request = self.identity.sign_request(request)

            # ─────────────────────────────────────────────────────────────
            # Execute handoff (in real implementation, this would send
            # to transport layer - for now we just prepare the result)
            # ─────────────────────────────────────────────────────────────

            duration_ms = (time.time() - start_time) * 1000

            result = HandoffResult(
                request_id=signed_request.request_id,
                status=HandoffStatus.COMPLETED,
                source_device=source_device,
                target_device=target_device,
                ops_transferred=len(pending_ops),
                duration_ms=duration_ms,
            )

            # Mark operations as synced
            self.memory.mark_synced(pending_ops)

            # Record success
            await self.circuit_breaker.record_success(target_device)

            # ─────────────────────────────────────────────────────────────
            # Layer 6 & 7: Log security event and audit trail
            # ─────────────────────────────────────────────────────────────
            self.monitor.log_event(SecurityEvent(
                event_type=SecurityEventType.HANDOFF_COMPLETE,
                timestamp=time.time(),
                device_id=target_device,
                identity_did=self.identity.did,
                details={
                    "ops_count": len(pending_ops),
                    "duration_ms": duration_ms,
                },
            ))

            self.audit.log(
                source_device=source_device,
                target_device=target_device,
                identity_did=self.identity.did,
                outcome="success",
                ops_count=len(pending_ops),
                duration_ms=duration_ms,
                risk_level=risk.level,
            )

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            # Record failure for circuit breaker
            await self.circuit_breaker.record_failure(target_device)

            # Log failure event
            self.monitor.log_event(SecurityEvent(
                event_type=SecurityEventType.HANDOFF_FAILED,
                timestamp=time.time(),
                device_id=target_device,
                identity_did=self.identity.did,
                details={"error": str(e)},
            ))

            # Audit failure
            self.audit.log(
                source_device=source_device,
                target_device=target_device,
                identity_did=self.identity.did,
                outcome=f"failed: {type(e).__name__}",
                ops_count=0,
                duration_ms=duration_ms,
                risk_level=RiskLevel.HIGH,
                details={"error": str(e)},
            )

            raise

    async def receive_handoff(
        self,
        signed_request: HandoffRequest,
    ) -> MergeResult:
        """
        Receive and validate incoming handoff from another device.

        Args:
            signed_request: Signed handoff request

        Returns:
            MergeResult with statistics

        Raises:
            InvalidSignatureError: Invalid signature
            ReplayAttackError: Replay attack detected
            DeviceNotFoundError: Source device unknown
            DeviceRevokedError: Source device revoked
        """
        start_time = time.time()

        # ─────────────────────────────────────────────────────────────
        # Layer 1: Verify source device
        # ─────────────────────────────────────────────────────────────
        if not self.identity.is_device_authorized(signed_request.source_device):
            device = self.identity.devices.get(signed_request.source_device)
            if device is None:
                self.monitor.log_event(SecurityEvent(
                    event_type=SecurityEventType.AUTH_FAILURE,
                    timestamp=time.time(),
                    device_id=signed_request.source_device,
                    identity_did=self.identity.did,
                    details={"reason": "unknown_device"},
                ))
                raise DeviceNotFoundError(
                    "Unknown source device",
                    device_id=signed_request.source_device,
                )
            else:
                raise DeviceRevokedError(
                    "Source device not authorized",
                    device_id=signed_request.source_device,
                )

        # ─────────────────────────────────────────────────────────────
        # Layer 1: Verify signature
        # ─────────────────────────────────────────────────────────────
        if not self.identity.verify_request(signed_request):
            self.monitor.log_event(SecurityEvent(
                event_type=SecurityEventType.AUTH_FAILURE,
                timestamp=time.time(),
                device_id=signed_request.source_device,
                identity_did=self.identity.did,
                details={"reason": "invalid_signature"},
            ))
            raise InvalidSignatureError(
                "Invalid handoff signature",
                device_id=signed_request.source_device,
            )

        # ─────────────────────────────────────────────────────────────
        # Layer 1: Check timestamp freshness (replay protection)
        # ─────────────────────────────────────────────────────────────
        age = time.time() - signed_request.timestamp
        if age > 300:  # 5 minute window
            self.monitor.log_event(SecurityEvent(
                event_type=SecurityEventType.AUTH_FAILURE,
                timestamp=time.time(),
                device_id=signed_request.source_device,
                identity_did=self.identity.did,
                details={"reason": "stale_timestamp", "age_seconds": age},
            ))
            raise ReplayAttackError(
                f"Handoff request too old: {age:.0f}s",
                device_id=signed_request.source_device,
            )

        # ─────────────────────────────────────────────────────────────
        # Merge incoming operations
        # ─────────────────────────────────────────────────────────────
        ops = [SignedOp.from_dict(op_dict) for op_dict in signed_request.pending_ops]
        result = await self.memory.merge(ops)

        duration_ms = (time.time() - start_time) * 1000

        # Log success
        self.monitor.log_event(SecurityEvent(
            event_type=SecurityEventType.AUTH_SUCCESS,
            timestamp=time.time(),
            device_id=signed_request.source_device,
            identity_did=self.identity.did,
            details={
                "merged": result.merged,
                "rejected": result.rejected,
            },
        ))

        self.audit.log(
            source_device=signed_request.source_device,
            target_device=self.identity.current_device_id or "unknown",
            identity_did=self.identity.did,
            outcome="receive_success",
            ops_count=result.merged,
            duration_ms=duration_ms,
            risk_level=RiskLevel.SAFE,
        )

        return result

    def get_security_stats(self) -> Dict[str, Any]:
        """Get comprehensive security statistics."""
        return {
            "recent_events": [
                e.to_dict() for e in self.monitor.get_recent_events(limit=20)
            ],
            "audit_entries": [
                e.to_dict() for e in self.audit.query(limit=20)
            ],
            "circuit_breakers": {
                device_id: {
                    "state": circuit.state.value,
                    "failure_count": circuit.failure_count,
                }
                for device_id, circuit in self.circuit_breaker._circuits.items()
            },
        }


# ═══════════════════════════════════════════════════════════════════════════
#  FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════


def create_hardened_orchestrator(
    identity: "UnifiedIdentity",
    memory: Optional[SyncedMemory] = None,
) -> HardenedHandoffOrchestrator:
    """
    Create a hardened handoff orchestrator with all security layers.

    Args:
        identity: UnifiedIdentity for signing
        memory: Optional SyncedMemory instance

    Returns:
        Configured HardenedHandoffOrchestrator
    """
    return HardenedHandoffOrchestrator(identity, memory)
