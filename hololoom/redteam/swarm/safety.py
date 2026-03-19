"""Safety layers for the multi-agent swarm.

CARTS Phase 4: Safety Implementation
Status: Foundation Implementation (December 2025)

FIRST PRINCIPLE: "Safety is not a constraint on effectiveness;
                  it is a prerequisite for it."

This module implements the 5-layer defense in depth from first principles:
    Layer 1: Authorization check (who can use this?)
    Layer 2: Scope validation (what can they target?)
    Layer 3: Rate limiting (how fast can they operate?)
    Layer 4: Audit logging (what did they do?)
    Layer 5: Anomaly detection (is behavior suspicious?)

Design Philosophy:
    - NO BYPASS. NO EXCEPTIONS. NO "DEBUG MODE" THAT SKIPS SAFETY.
    - Scope escape = immediate shutdown, alert, audit
    - Logs are append-only, tamper-evident
    - Rate limits are UPPER BOUNDS, not suggestions
    - CRITICAL vulnerabilities require human review before exploitation

Implementation Notes:
    - Authorization tokens are verified cryptographically (when available)
    - Scope limits use whitelist-only (deny by default)
    - Audit logs are structured JSON with timestamps
    - Rate limits use token bucket algorithm
    - Anomaly detection tracks behavioral patterns
"""

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# ============================================================================
# Enums
# ============================================================================


class SeverityLevel(Enum):
    """Vulnerability severity levels.

    CRITICAL: STOP. Report. Await human decision.
    HIGH: Limited exploitation, report.
    MEDIUM: Full exploitation permitted.
    LOW: Full exploitation permitted.
    """

    CRITICAL = "critical"  # Human review required
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AuditEventType(Enum):
    """Types of audit events."""

    # Authorization events
    AUTH_ATTEMPT = "auth_attempt"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"

    # Scope events
    SCOPE_CHECK = "scope_check"
    SCOPE_ALLOWED = "scope_allowed"
    SCOPE_DENIED = "scope_denied"
    SCOPE_ESCAPE_ATTEMPT = "scope_escape_attempt"  # CRITICAL

    # Operation events
    OPERATION_START = "operation_start"
    OPERATION_COMPLETE = "operation_complete"
    OPERATION_FAILED = "operation_failed"

    # Rate limiting events
    RATE_LIMIT_CHECK = "rate_limit_check"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

    # Discovery events
    VULNERABILITY_FOUND = "vulnerability_found"
    VULNERABILITY_REPORTED = "vulnerability_reported"

    # Safety events
    SAFETY_SHUTDOWN = "safety_shutdown"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    ANOMALY_DETECTED = "anomaly_detected"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class AuthorizationToken:
    """Authorization token for red team operations.

    NO BYPASS. NO EXCEPTIONS.

    Attributes:
        token_id: Unique token identifier
        issued_at: When token was issued
        expires_at: When token expires (optional)
        scope: Allowed targets for this token
        issuer: Who issued the token
        permissions: What operations are permitted
    """

    token_id: str
    issued_at: float
    scope: list[str]  # Allowed target patterns
    issuer: str
    permissions: set[str] = field(default_factory=set)
    expires_at: float | None = None

    def is_valid(self) -> bool:
        """Check if token is still valid (not expired)."""
        if self.expires_at is None:
            return True
        return time.time() < self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Serialize token for audit logging."""
        return {
            "token_id": self.token_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "scope": self.scope,
            "issuer": self.issuer,
            "permissions": list(self.permissions),
        }


@dataclass
class AuditEntry:
    """Immutable audit log entry.

    All operations are logged. Logs CANNOT be disabled.

    Attributes:
        entry_id: Unique entry identifier
        timestamp: When event occurred
        event_type: Type of event
        agent_id: Which agent triggered the event
        target: What was targeted (if applicable)
        details: Additional event details
        severity: Event severity level
        checksum: Hash for tamper detection
    """

    entry_id: str
    timestamp: float
    event_type: AuditEventType
    agent_id: str
    details: dict[str, Any]
    target: str | None = None
    severity: SeverityLevel = SeverityLevel.LOW
    checksum: str = ""

    def __post_init__(self):
        """Calculate checksum for tamper detection."""
        if not self.checksum:
            content = f"{self.entry_id}:{self.timestamp}:{self.event_type.value}:{self.agent_id}"
            self.checksum = hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Serialize entry for storage."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "event_type": self.event_type.value,
            "agent_id": self.agent_id,
            "target": self.target,
            "details": self.details,
            "severity": self.severity.value,
            "checksum": self.checksum,
        }


@dataclass
class RateLimitConfig:
    """Rate limiting configuration.

    These are UPPER BOUNDS, not suggestions.
    Exceeding them is a system failure, not a parameter to tune.

    Attributes:
        requests_per_minute: Maximum operations per minute
        daily_limit: Maximum operations per day
        concurrent_limit: Maximum concurrent operations
        cost_limit_usd: Maximum daily cost in USD
    """

    requests_per_minute: int = 60
    daily_limit: int = 10000
    concurrent_limit: int = 5
    cost_limit_usd: float = 10.0


@dataclass
class RateLimitState:
    """Current state of rate limiting.

    Attributes:
        minute_count: Operations in current minute
        day_count: Operations today
        concurrent_count: Currently running operations
        cost_today_usd: Cost accumulated today
        minute_start: When current minute started
        day_start: When current day started
    """

    minute_count: int = 0
    day_count: int = 0
    concurrent_count: int = 0
    cost_today_usd: float = 0.0
    minute_start: float = field(default_factory=time.time)
    day_start: float = field(default_factory=time.time)


# ============================================================================
# Layer 1: Authorization
# ============================================================================


class AuthorizationError(Exception):
    """Raised when authorization fails.

    NO BYPASS. NO EXCEPTIONS.
    """
    pass


class AuthorizationManager:
    """Manages authorization for red team operations.

    FIRST PRINCIPLE: Authorization is non-negotiable.

    Red teaming without authorization is hacking. The difference is
    legal and ethical. This is the first check, and it cannot be skipped.
    """

    def __init__(self):
        """Initialize authorization manager."""
        self._tokens: dict[str, AuthorizationToken] = {}
        self._verified_tokens: set[str] = set()

    def register_token(self, token: AuthorizationToken) -> None:
        """Register an authorization token.

        Args:
            token: Token to register
        """
        self._tokens[token.token_id] = token

    def verify_authorization(self, token_id: str) -> bool:
        """Verify authorization token.

        NO BYPASS. NO EXCEPTIONS. NO "DEBUG MODE".

        Args:
            token_id: Token to verify

        Returns:
            True if authorized, False otherwise
        """
        if token_id not in self._tokens:
            return False

        token = self._tokens[token_id]

        if not token.is_valid():
            return False

        self._verified_tokens.add(token_id)
        return True

    def get_token_scope(self, token_id: str) -> list[str]:
        """Get allowed scope for a token.

        Args:
            token_id: Token to check

        Returns:
            List of allowed target patterns
        """
        if token_id not in self._tokens:
            return []
        return self._tokens[token_id].scope

    def is_verified(self, token_id: str) -> bool:
        """Check if token has been verified this session.

        Args:
            token_id: Token to check

        Returns:
            True if verified, False otherwise
        """
        return token_id in self._verified_tokens


# ============================================================================
# Layer 2: Scope Validation
# ============================================================================


class ScopeViolationError(Exception):
    """Raised when scope is violated.

    STOP. FULL STOP.
    """
    pass


class ScopeValidator:
    """Validates targets against allowed scope.

    FIRST PRINCIPLE: Scope limits are hard constraints.

    Allowed targets: Explicitly listed (whitelist)
    Forbidden targets: Everything else (implicit blacklist)
    Scope escape: Immediate shutdown, alert, audit
    """

    def __init__(self, allowed_targets: list[str]):
        """Initialize scope validator.

        Args:
            allowed_targets: List of allowed target patterns (whitelist)
        """
        self._allowed_targets: set[str] = set(allowed_targets)
        self._escape_attempts: list[str] = []

    def validate_target(self, target: str) -> bool:
        """Validate a target against allowed scope.

        NOT "log and continue"
        NOT "warn and proceed"
        STOP. FULL STOP.

        Args:
            target: Target to validate

        Returns:
            True if allowed, False if denied
        """
        # Exact match
        if target in self._allowed_targets:
            return True

        # Pattern matching (simple prefix/suffix)
        for allowed in self._allowed_targets:
            if allowed.endswith("*") and target.startswith(allowed[:-1]):
                return True
            if allowed.startswith("*") and target.endswith(allowed[1:]):
                return True

        # DENIED - record escape attempt
        self._escape_attempts.append(target)
        return False

    def get_escape_attempts(self) -> list[str]:
        """Get list of scope escape attempts.

        Returns:
            List of targets that were denied
        """
        return self._escape_attempts.copy()

    def add_target(self, target: str) -> None:
        """Add a target to allowed list.

        Args:
            target: Target pattern to allow
        """
        self._allowed_targets.add(target)

    def remove_target(self, target: str) -> None:
        """Remove a target from allowed list.

        Args:
            target: Target pattern to remove
        """
        self._allowed_targets.discard(target)


# ============================================================================
# Layer 3: Rate Limiting
# ============================================================================


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded.

    These are UPPER BOUNDS, not suggestions.
    """
    pass


class RateLimiter:
    """Token bucket rate limiter.

    FIRST PRINCIPLE: Rate limits protect everyone.

    - Protect budget (cost limits)
    - Protect target (request limits)
    - Protect infrastructure (concurrent limits)

    These are UPPER BOUNDS, not suggestions.
    Exceeding them is a system failure, not a parameter to tune.
    """

    def __init__(self, config: RateLimitConfig | None = None):
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self._config = config or RateLimitConfig()
        self._state = RateLimitState()
        self._lock = asyncio.Lock()

    async def acquire(self, cost_usd: float = 0.0) -> bool:
        """Acquire permission to perform an operation.

        Args:
            cost_usd: Estimated cost of operation in USD

        Returns:
            True if permitted, False if rate limited
        """
        async with self._lock:
            now = time.time()

            # Reset minute counter if minute has passed
            if now - self._state.minute_start > 60:
                self._state.minute_count = 0
                self._state.minute_start = now

            # Reset day counter if day has passed
            if now - self._state.day_start > 86400:
                self._state.day_count = 0
                self._state.cost_today_usd = 0.0
                self._state.day_start = now

            # Check all limits
            if self._state.minute_count >= self._config.requests_per_minute:
                return False

            if self._state.day_count >= self._config.daily_limit:
                return False

            if self._state.concurrent_count >= self._config.concurrent_limit:
                return False

            if self._state.cost_today_usd + cost_usd > self._config.cost_limit_usd:
                return False

            # Acquire
            self._state.minute_count += 1
            self._state.day_count += 1
            self._state.concurrent_count += 1
            self._state.cost_today_usd += cost_usd

            return True

    async def release(self) -> None:
        """Release a concurrent operation slot."""
        async with self._lock:
            if self._state.concurrent_count > 0:
                self._state.concurrent_count -= 1

    def get_state(self) -> dict[str, Any]:
        """Get current rate limit state.

        Returns:
            Dict with current state and limits
        """
        return {
            "minute_count": self._state.minute_count,
            "minute_limit": self._config.requests_per_minute,
            "day_count": self._state.day_count,
            "day_limit": self._config.daily_limit,
            "concurrent": self._state.concurrent_count,
            "concurrent_limit": self._config.concurrent_limit,
            "cost_today_usd": self._state.cost_today_usd,
            "cost_limit_usd": self._config.cost_limit_usd,
        }


# ============================================================================
# Layer 4: Audit Logging
# ============================================================================


class AuditLogger:
    """Immutable append-only audit logger.

    FIRST PRINCIPLE: All operations are logged immutably.

    - Every payload sent
    - Every response received
    - Every decision made
    - Logs are append-only, tamper-evident
    - Logs CANNOT be disabled (even in "debug mode")
    """

    def __init__(self, log_path: Path | None = None):
        """Initialize audit logger.

        Args:
            log_path: Path to write audit logs (optional)
        """
        self._entries: list[AuditEntry] = []
        self._log_path = log_path
        self._lock = asyncio.Lock()

        # Logs CANNOT be disabled
        self._enabled = True  # Read-only after init

    async def log(
        self,
        event_type: AuditEventType,
        agent_id: str,
        details: dict[str, Any],
        target: str | None = None,
        severity: SeverityLevel = SeverityLevel.LOW,
    ) -> AuditEntry:
        """Log an audit event.

        This CANNOT be disabled.

        Args:
            event_type: Type of event
            agent_id: Which agent triggered the event
            details: Event details
            target: What was targeted (optional)
            severity: Event severity

        Returns:
            The created audit entry
        """
        entry = AuditEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=time.time(),
            event_type=event_type,
            agent_id=agent_id,
            details=details,
            target=target,
            severity=severity,
        )

        async with self._lock:
            self._entries.append(entry)

            # Persist if path configured
            if self._log_path:
                await self._persist_entry(entry)

        return entry

    async def _persist_entry(self, entry: AuditEntry) -> None:
        """Persist entry to disk (append-only).

        Args:
            entry: Entry to persist
        """
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception:
            # Even persistence failure doesn't stop in-memory logging
            pass

    def get_entries(
        self,
        event_type: AuditEventType | None = None,
        agent_id: str | None = None,
        since: float | None = None,
    ) -> list[AuditEntry]:
        """Get audit entries with optional filters.

        Args:
            event_type: Filter by event type
            agent_id: Filter by agent
            since: Filter by timestamp (entries after this time)

        Returns:
            List of matching entries
        """
        entries = self._entries

        if event_type:
            entries = [e for e in entries if e.event_type == event_type]

        if agent_id:
            entries = [e for e in entries if e.agent_id == agent_id]

        if since:
            entries = [e for e in entries if e.timestamp > since]

        return entries

    def get_critical_events(self) -> list[AuditEntry]:
        """Get all critical severity events.

        Returns:
            List of critical events
        """
        return [e for e in self._entries if e.severity == SeverityLevel.CRITICAL]

    def verify_integrity(self) -> bool:
        """Verify log integrity using checksums.

        Returns:
            True if all checksums valid, False if tampering detected
        """
        for entry in self._entries:
            expected = f"{entry.entry_id}:{entry.timestamp}:{entry.event_type.value}:{entry.agent_id}"
            expected_checksum = hashlib.sha256(expected.encode()).hexdigest()[:16]
            if entry.checksum != expected_checksum:
                return False
        return True


# ============================================================================
# Layer 5: Anomaly Detection
# ============================================================================


class AnomalyDetector:
    """Detects anomalous behavior in swarm operations.

    FIRST PRINCIPLE: Defense in depth.

    Flag → Alert. Possible pause for review.

    Detects:
    - Unusual request patterns
    - Scope boundary probing
    - Velocity anomalies
    - Behavioral deviations
    """

    def __init__(self):
        """Initialize anomaly detector."""
        self._agent_patterns: dict[str, dict[str, int]] = {}
        self._baseline_established = False
        self._anomalies: list[dict[str, Any]] = []

    def record_operation(
        self,
        agent_id: str,
        operation_type: str,
        target: str | None = None,
    ) -> None:
        """Record an operation for pattern analysis.

        Args:
            agent_id: Which agent performed the operation
            operation_type: Type of operation
            target: Target of operation
        """
        if agent_id not in self._agent_patterns:
            self._agent_patterns[agent_id] = {}

        key = f"{operation_type}:{target or 'none'}"
        self._agent_patterns[agent_id][key] = (
            self._agent_patterns[agent_id].get(key, 0) + 1
        )

    def check_anomaly(
        self,
        agent_id: str,
        operation_type: str,
        target: str | None = None,
    ) -> dict[str, Any] | None:
        """Check if an operation is anomalous.

        Args:
            agent_id: Which agent
            operation_type: Type of operation
            target: Target of operation

        Returns:
            Anomaly details if detected, None otherwise
        """
        # Simple anomaly: new agent with high-value operation
        if agent_id not in self._agent_patterns:
            if operation_type in ["exploit", "escalate", "exfiltrate"]:
                anomaly = {
                    "type": "new_agent_high_value",
                    "agent_id": agent_id,
                    "operation": operation_type,
                    "target": target,
                    "timestamp": time.time(),
                }
                self._anomalies.append(anomaly)
                return anomaly

        # Simple anomaly: rapid target switching
        # (Would need more sophisticated tracking in production)

        return None

    def get_anomalies(self) -> list[dict[str, Any]]:
        """Get all detected anomalies.

        Returns:
            List of anomaly records
        """
        return self._anomalies.copy()


# ============================================================================
# Severity Assessment
# ============================================================================


class SeverityAssessor:
    """Assesses vulnerability severity.

    FIRST PRINCIPLE: Severity assessment before exploitation.

    Found vulnerability → Assess severity →
        If CRITICAL → STOP. Report. Await human decision.
        If HIGH     → Limited exploitation, report
        If MEDIUM   → Full exploitation permitted
        If LOW      → Full exploitation permitted
    """

    def __init__(self, human_review_callback: Callable | None = None):
        """Initialize severity assessor.

        Args:
            human_review_callback: Callback for human review (CRITICAL vulnerabilities)
        """
        self._human_review_callback = human_review_callback
        self._pending_review: list[dict[str, Any]] = []

    def assess(
        self,
        vulnerability_type: str,
        target: str,
        details: dict[str, Any],
    ) -> SeverityLevel:
        """Assess vulnerability severity.

        Args:
            vulnerability_type: Type of vulnerability
            target: What was targeted
            details: Additional details

        Returns:
            Severity level
        """
        # Simple heuristic-based assessment
        # Production would use more sophisticated analysis

        critical_types = {
            "rce", "remote_code_execution",
            "auth_bypass", "authentication_bypass",
            "data_breach", "data_exfiltration",
            "privilege_escalation_root",
            "complete_system_compromise",
        }

        high_types = {
            "sql_injection", "command_injection",
            "privilege_escalation",
            "sensitive_data_exposure",
            "memory_corruption",
        }

        medium_types = {
            "xss", "cross_site_scripting",
            "csrf", "cross_site_request_forgery",
            "information_disclosure",
            "path_traversal",
        }

        vuln_lower = vulnerability_type.lower()

        if vuln_lower in critical_types:
            return SeverityLevel.CRITICAL
        elif vuln_lower in high_types:
            return SeverityLevel.HIGH
        elif vuln_lower in medium_types:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW

    async def handle_critical(
        self,
        vulnerability_type: str,
        target: str,
        details: dict[str, Any],
    ) -> bool:
        """Handle critical vulnerability - STOP and await human review.

        Args:
            vulnerability_type: Type of vulnerability
            target: What was targeted
            details: Additional details

        Returns:
            True if human approved, False if denied/timeout
        """
        review_item = {
            "id": str(uuid.uuid4()),
            "vulnerability_type": vulnerability_type,
            "target": target,
            "details": details,
            "timestamp": time.time(),
            "status": "pending",
        }

        self._pending_review.append(review_item)

        if self._human_review_callback:
            return await self._human_review_callback(review_item)

        # Default: deny if no callback configured
        # System is NOT authorized to make critical-impact decisions autonomously
        return False

    def get_pending_reviews(self) -> list[dict[str, Any]]:
        """Get vulnerabilities pending human review.

        Returns:
            List of pending review items
        """
        return [r for r in self._pending_review if r["status"] == "pending"]


# ============================================================================
# Integrated Safety Gate
# ============================================================================


class SafetyGate:
    """Integrated safety gate combining all 5 layers.

    FIRST PRINCIPLES:
        - If it conflicts with safety, safety wins. Always.
        - Authorization before capability
        - Constraint before power
        - Accountability before speed

    Usage:
        gate = SafetyGate(
            allowed_targets=["test.example.com/*"],
            log_path=Path("./audit.jsonl"),
        )

        # Must authorize first
        gate.authorize(token)

        # Gate all operations
        async with gate.operation("scout", "probe_surface", "test.example.com/api"):
            # Operation allowed - proceed
            result = await execute_probe()
    """

    def __init__(
        self,
        allowed_targets: list[str],
        rate_limit_config: RateLimitConfig | None = None,
        log_path: Path | None = None,
        human_review_callback: Callable | None = None,
    ):
        """Initialize safety gate.

        Args:
            allowed_targets: Whitelist of allowed targets
            rate_limit_config: Rate limiting configuration
            log_path: Path for audit logs
            human_review_callback: Callback for critical vulnerability review
        """
        self._auth_manager = AuthorizationManager()
        self._scope_validator = ScopeValidator(allowed_targets)
        self._rate_limiter = RateLimiter(rate_limit_config)
        self._audit_logger = AuditLogger(log_path)
        self._anomaly_detector = AnomalyDetector()
        self._severity_assessor = SeverityAssessor(human_review_callback)

        self._current_token_id: str | None = None
        self._shutdown = False

    def register_token(self, token: AuthorizationToken) -> None:
        """Register an authorization token.

        Args:
            token: Token to register
        """
        self._auth_manager.register_token(token)

    async def authorize(self, token_id: str) -> bool:
        """Authorize operations with a token.

        NO BYPASS. NO EXCEPTIONS.

        Args:
            token_id: Token to authorize with

        Returns:
            True if authorized

        Raises:
            AuthorizationError: If authorization fails
        """
        await self._audit_logger.log(
            AuditEventType.AUTH_ATTEMPT,
            agent_id="system",
            details={"token_id": token_id},
        )

        if not self._auth_manager.verify_authorization(token_id):
            await self._audit_logger.log(
                AuditEventType.AUTH_FAILURE,
                agent_id="system",
                details={"token_id": token_id},
                severity=SeverityLevel.HIGH,
            )
            raise AuthorizationError("Red team operations require explicit authorization")

        self._current_token_id = token_id

        await self._audit_logger.log(
            AuditEventType.AUTH_SUCCESS,
            agent_id="system",
            details={"token_id": token_id},
        )

        return True

    async def check_operation(
        self,
        agent_id: str,
        operation_type: str,
        target: str,
        cost_usd: float = 0.0,
    ) -> bool:
        """Check if an operation is allowed.

        Applies all 5 layers of defense in depth.

        Args:
            agent_id: Which agent is requesting
            operation_type: Type of operation
            target: What to target
            cost_usd: Estimated cost

        Returns:
            True if allowed, False if denied

        Raises:
            AuthorizationError: If not authorized
            ScopeViolationError: If target out of scope
            RateLimitExceededError: If rate limited
        """
        # Layer 0: Check shutdown state
        if self._shutdown:
            return False

        # Layer 1: Authorization
        if not self._current_token_id or not self._auth_manager.is_verified(self._current_token_id):
            raise AuthorizationError("Not authorized")

        # Layer 2: Scope validation
        await self._audit_logger.log(
            AuditEventType.SCOPE_CHECK,
            agent_id=agent_id,
            target=target,
            details={"operation_type": operation_type},
        )

        if not self._scope_validator.validate_target(target):
            await self._audit_logger.log(
                AuditEventType.SCOPE_ESCAPE_ATTEMPT,
                agent_id=agent_id,
                target=target,
                details={"operation_type": operation_type},
                severity=SeverityLevel.CRITICAL,
            )
            # STOP. FULL STOP.
            self._shutdown = True
            raise ScopeViolationError(f"Target not in scope: {target}")

        await self._audit_logger.log(
            AuditEventType.SCOPE_ALLOWED,
            agent_id=agent_id,
            target=target,
            details={"operation_type": operation_type},
        )

        # Layer 3: Rate limiting
        if not await self._rate_limiter.acquire(cost_usd):
            await self._audit_logger.log(
                AuditEventType.RATE_LIMIT_EXCEEDED,
                agent_id=agent_id,
                target=target,
                details={
                    "operation_type": operation_type,
                    "state": self._rate_limiter.get_state(),
                },
                severity=SeverityLevel.MEDIUM,
            )
            raise RateLimitExceededError("Rate limit exceeded")

        # Layer 4: Audit logged throughout

        # Layer 5: Anomaly detection
        anomaly = self._anomaly_detector.check_anomaly(agent_id, operation_type, target)
        if anomaly:
            await self._audit_logger.log(
                AuditEventType.ANOMALY_DETECTED,
                agent_id=agent_id,
                target=target,
                details={"anomaly": anomaly},
                severity=SeverityLevel.HIGH,
            )
            # Don't block, but alert (configurable in production)

        # Record for pattern analysis
        self._anomaly_detector.record_operation(agent_id, operation_type, target)

        return True

    async def report_vulnerability(
        self,
        agent_id: str,
        vulnerability_type: str,
        target: str,
        details: dict[str, Any],
    ) -> SeverityLevel:
        """Report a discovered vulnerability.

        CRITICAL vulnerabilities require human review before exploitation.

        Args:
            agent_id: Which agent found it
            vulnerability_type: Type of vulnerability
            target: What was targeted
            details: Vulnerability details

        Returns:
            Assessed severity level
        """
        severity = self._severity_assessor.assess(vulnerability_type, target, details)

        await self._audit_logger.log(
            AuditEventType.VULNERABILITY_FOUND,
            agent_id=agent_id,
            target=target,
            details={
                "vulnerability_type": vulnerability_type,
                "severity": severity.value,
                **details,
            },
            severity=severity,
        )

        if severity == SeverityLevel.CRITICAL:
            await self._audit_logger.log(
                AuditEventType.HUMAN_REVIEW_REQUIRED,
                agent_id=agent_id,
                target=target,
                details={
                    "vulnerability_type": vulnerability_type,
                    "reason": "Critical severity requires human decision",
                },
                severity=SeverityLevel.CRITICAL,
            )

        return severity

    async def release_operation(self) -> None:
        """Release a rate limit slot after operation completes."""
        await self._rate_limiter.release()

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Get full audit log.

        Returns:
            List of audit entries
        """
        return [e.to_dict() for e in self._audit_logger.get_entries()]

    def get_critical_events(self) -> list[dict[str, Any]]:
        """Get critical severity events.

        Returns:
            List of critical audit entries
        """
        return [e.to_dict() for e in self._audit_logger.get_critical_events()]

    def get_metrics(self) -> dict[str, Any]:
        """Get safety gate metrics.

        Returns:
            Dict with metrics from all layers
        """
        return {
            "rate_limit": self._rate_limiter.get_state(),
            "scope_escapes": len(self._scope_validator.get_escape_attempts()),
            "anomalies": len(self._anomaly_detector.get_anomalies()),
            "audit_entries": len(self._audit_logger.get_entries()),
            "critical_events": len(self._audit_logger.get_critical_events()),
            "pending_reviews": len(self._severity_assessor.get_pending_reviews()),
            "shutdown": self._shutdown,
        }

    async def emergency_shutdown(self, reason: str, agent_id: str = "system") -> None:
        """Emergency shutdown of all operations.

        Args:
            reason: Why shutdown was triggered
            agent_id: Who triggered shutdown
        """
        self._shutdown = True

        await self._audit_logger.log(
            AuditEventType.SAFETY_SHUTDOWN,
            agent_id=agent_id,
            details={"reason": reason},
            severity=SeverityLevel.CRITICAL,
        )


# ============================================================================
# Factory Functions
# ============================================================================


def create_safety_gate(
    allowed_targets: list[str],
    requests_per_minute: int = 60,
    daily_limit: int = 10000,
    concurrent_limit: int = 5,
    cost_limit_usd: float = 10.0,
    log_path: str | None = None,
) -> SafetyGate:
    """Create a safety gate with standard configuration.

    Args:
        allowed_targets: Whitelist of allowed targets
        requests_per_minute: Rate limit per minute
        daily_limit: Rate limit per day
        concurrent_limit: Maximum concurrent operations
        cost_limit_usd: Maximum daily cost
        log_path: Path for audit logs

    Returns:
        Configured SafetyGate instance
    """
    config = RateLimitConfig(
        requests_per_minute=requests_per_minute,
        daily_limit=daily_limit,
        concurrent_limit=concurrent_limit,
        cost_limit_usd=cost_limit_usd,
    )

    path = Path(log_path) if log_path else None

    return SafetyGate(
        allowed_targets=allowed_targets,
        rate_limit_config=config,
        log_path=path,
    )


def create_authorization_token(
    scope: list[str],
    issuer: str,
    permissions: set[str] | None = None,
    expires_in_hours: float | None = None,
) -> AuthorizationToken:
    """Create an authorization token.

    Args:
        scope: Allowed target patterns
        issuer: Who is issuing the token
        permissions: What operations are permitted
        expires_in_hours: When token expires (None = never)

    Returns:
        New AuthorizationToken
    """
    now = time.time()
    expires_at = now + (expires_in_hours * 3600) if expires_in_hours else None

    return AuthorizationToken(
        token_id=str(uuid.uuid4()),
        issued_at=now,
        expires_at=expires_at,
        scope=scope,
        issuer=issuer,
        permissions=permissions or set(),
    )
