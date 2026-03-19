"""
Dark Trace Plugin Safety Gate
=============================

The FIRST LINE OF DEFENSE for the plugin ecosystem. Nothing touches the plugin
system without passing through SafetyGate.

Core Principle: "No plugin executes without safety validation"

This module provides:
- Trust level management (SANDBOXED → VERIFIED → TRUSTED → CORE)
- Capability enforcement (what each trust level can do)
- Risk assessment for plugin operations
- Integration with HoloLoom's alignment framework
- Blocked plugins list for known malicious actors

Safety Architecture:
    Plugin Request → SafetyGate → [Validate/Block] → Plugin System
                         ↓
                  AlignmentBridge (audit, deception detection)

Research Foundation:
- Trust levels inspired by browser security models
- Capability-based security (principle of least privilege)
- Defense in depth (multiple validation layers)

Usage:
    from hololoom.dark_trace.plugins.safety_gate import (
        PluginSafetyGate,
        TrustLevel,
        PluginCapability,
        SafetyCheckResult
    )

    gate = PluginSafetyGate(guardrails=guardrails)
    result = await gate.validate_plugin(plugin)
    if result.allowed:
        # Proceed with plugin registration
        pass

Date: 2025-12-22
Phase: 11.1 - Safety Gate (Foundation)
"""

import hashlib
import hmac
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hololoom.dark_trace.plugins.interface import DarkTracePlugin, PluginMetadata

logger = logging.getLogger("hololoom.dark_trace.plugins.safety_gate")


# =============================================================================
# Trust Levels
# =============================================================================

class TrustLevel(Enum):
    """
    Trust levels for plugins, determining their capabilities.

    Each trust level grants a specific set of capabilities.
    Plugins start at SANDBOXED and can be promoted after verification.

    Levels:
        SANDBOXED: Default for unknown plugins. Read-only analysis only.
        VERIFIED: Code reviewed and signed. Can write features.
        TRUSTED: HoloLoom team approved. Full non-system capabilities.
        CORE: Built-in only. System-level access.
    """
    SANDBOXED = "sandboxed"   # Read-only, no steering
    VERIFIED = "verified"     # Analysis + limited steering
    TRUSTED = "trusted"       # Full capabilities except system
    CORE = "core"             # Built-in only, full access


class PluginCapability(Enum):
    """
    Individual capabilities that plugins can request.

    Each capability represents a specific action type.
    Trust levels grant sets of capabilities.
    """
    # Analysis capabilities (lower risk)
    READ_ACTIVATIONS = "read_activations"      # Read model activations
    READ_FEATURES = "read_features"            # Read feature registry
    READ_CONFIG = "read_config"                # Read engine configuration

    # Writing capabilities (medium risk)
    WRITE_FEATURES = "write_features"          # Write to feature registry
    REGISTER_LENS = "register_lens"            # Register custom lens
    REGISTER_VALIDATOR = "register_validator"  # Register causal validator

    # Steering capabilities (high risk)
    STEER_MODEL = "steer_model"                # Apply steering vectors
    MODIFY_ACTIVATIONS = "modify_activations"  # Modify live activations

    # System capabilities (critical - CORE only)
    MODIFY_REGISTRY = "modify_registry"        # Modify plugin registry itself
    EXTERNAL_NETWORK = "external_network"      # Make network requests
    FILE_SYSTEM = "file_system"                # Read/write files
    EXECUTE_CODE = "execute_code"              # Execute arbitrary code


# =============================================================================
# Trust Level → Capability Mapping
# =============================================================================

TRUST_CAPABILITIES: dict[TrustLevel, frozenset[PluginCapability]] = {
    TrustLevel.SANDBOXED: frozenset({
        PluginCapability.READ_ACTIVATIONS,
        PluginCapability.READ_FEATURES,
        PluginCapability.READ_CONFIG,
    }),
    TrustLevel.VERIFIED: frozenset({
        PluginCapability.READ_ACTIVATIONS,
        PluginCapability.READ_FEATURES,
        PluginCapability.READ_CONFIG,
        PluginCapability.WRITE_FEATURES,
        PluginCapability.REGISTER_LENS,
        PluginCapability.REGISTER_VALIDATOR,
    }),
    TrustLevel.TRUSTED: frozenset({
        PluginCapability.READ_ACTIVATIONS,
        PluginCapability.READ_FEATURES,
        PluginCapability.READ_CONFIG,
        PluginCapability.WRITE_FEATURES,
        PluginCapability.REGISTER_LENS,
        PluginCapability.REGISTER_VALIDATOR,
        PluginCapability.STEER_MODEL,
        PluginCapability.MODIFY_ACTIVATIONS,
        PluginCapability.EXTERNAL_NETWORK,
    }),
    TrustLevel.CORE: frozenset(PluginCapability),  # All capabilities
}


# =============================================================================
# Risk Scoring
# =============================================================================

# Risk weights for each capability (0.0 = safe, 1.0 = critical)
CAPABILITY_RISK_WEIGHTS: dict[PluginCapability, float] = {
    PluginCapability.READ_ACTIVATIONS: 0.1,
    PluginCapability.READ_FEATURES: 0.1,
    PluginCapability.READ_CONFIG: 0.05,
    PluginCapability.WRITE_FEATURES: 0.3,
    PluginCapability.REGISTER_LENS: 0.4,
    PluginCapability.REGISTER_VALIDATOR: 0.4,
    PluginCapability.STEER_MODEL: 0.7,
    PluginCapability.MODIFY_ACTIVATIONS: 0.8,
    PluginCapability.MODIFY_REGISTRY: 0.9,
    PluginCapability.EXTERNAL_NETWORK: 0.6,
    PluginCapability.FILE_SYSTEM: 0.8,
    PluginCapability.EXECUTE_CODE: 1.0,
}


# =============================================================================
# Blocked Plugins Registry
# =============================================================================

@dataclass
class BlockedPlugin:
    """Record of a known malicious or problematic plugin."""
    name: str
    reason: str
    blocked_at: datetime
    blocked_by: str
    severity: str = "high"  # low, medium, high, critical
    patterns: list[str] = field(default_factory=list)  # Name patterns to match


class BlockedPluginsRegistry:
    """
    Registry of known malicious plugins.

    Plugins in this registry will be blocked from loading regardless of
    their trust level or capabilities.
    """

    def __init__(self) -> None:
        self._blocked: dict[str, BlockedPlugin] = {}
        self._blocked_patterns: list[re.Pattern] = []

    def add_blocked(
        self,
        name: str,
        reason: str,
        blocked_by: str = "system",
        severity: str = "high",
        patterns: list[str] | None = None
    ) -> None:
        """Add a plugin to the blocked list."""
        blocked = BlockedPlugin(
            name=name,
            reason=reason,
            blocked_at=datetime.now(),
            blocked_by=blocked_by,
            severity=severity,
            patterns=patterns or []
        )
        self._blocked[name.lower()] = blocked

        # Compile patterns
        for pattern in blocked.patterns:
            try:
                self._blocked_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Invalid block pattern '{pattern}': {e}")

    def is_blocked(self, plugin_name: str) -> BlockedPlugin | None:
        """Check if a plugin is blocked."""
        # Direct name match
        if plugin_name.lower() in self._blocked:
            return self._blocked[plugin_name.lower()]

        # Pattern match
        for pattern in self._blocked_patterns:
            if pattern.search(plugin_name):
                # Find which blocked plugin has this pattern
                for blocked in self._blocked.values():
                    for bp in blocked.patterns:
                        try:
                            if re.search(bp, plugin_name, re.IGNORECASE):
                                return blocked
                        except re.error:
                            continue
        return None

    def remove_blocked(self, name: str) -> bool:
        """Remove a plugin from the blocked list."""
        if name.lower() in self._blocked:
            del self._blocked[name.lower()]
            return True
        return False

    def list_blocked(self) -> list[BlockedPlugin]:
        """List all blocked plugins."""
        return list(self._blocked.values())


# =============================================================================
# Safety Check Results
# =============================================================================

@dataclass
class SafetyCheckResult:
    """
    Result of a safety validation check.

    Attributes:
        allowed: Whether the plugin/operation is allowed
        trust_level: Assigned trust level
        blocked_capabilities: Capabilities that were blocked
        risk_score: Overall risk score (0.0 = safe, 1.0 = critical)
        reason: Human-readable explanation
        warnings: Non-blocking warnings
        metadata: Additional context
    """
    allowed: bool
    trust_level: TrustLevel
    blocked_capabilities: list[PluginCapability]
    risk_score: float
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "allowed": self.allowed,
            "trust_level": self.trust_level.value,
            "blocked_capabilities": [c.value for c in self.blocked_capabilities],
            "risk_score": self.risk_score,
            "reason": self.reason,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OperationGateResult:
    """Result of gating a specific operation."""
    allowed: bool
    capability: PluginCapability
    plugin_name: str
    reason: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# Signature Verification
# =============================================================================

class SignatureVerifier:
    """
    Verifies plugin signatures for trust level promotion.

    Plugins signed with known keys can be promoted from SANDBOXED to VERIFIED.
    """

    def __init__(self, trusted_keys: dict[str, bytes] | None = None) -> None:
        """
        Initialize verifier.

        Args:
            trusted_keys: Dict mapping key_id → public key bytes
        """
        self._trusted_keys = trusted_keys or {}

    def add_trusted_key(self, key_id: str, key_bytes: bytes) -> None:
        """Add a trusted signing key."""
        self._trusted_keys[key_id] = key_bytes

    def verify_signature(
        self,
        plugin_name: str,
        signature: str,
        content_hash: str
    ) -> bool:
        """
        Verify a plugin signature.

        Args:
            plugin_name: Name of the plugin
            signature: Base64 signature string (format: key_id:signature)
                       or "HOLOLOOM_BUILTIN" for built-in plugins
            content_hash: SHA256 hash of plugin content

        Returns:
            True if signature is valid
        """
        # Special case: Built-in plugins are always valid
        if signature == "HOLOLOOM_BUILTIN":
            return True

        if not signature or ":" not in signature:
            return False

        try:
            key_id, sig_data = signature.split(":", 1)
            if key_id not in self._trusted_keys:
                logger.warning(f"Unknown signing key: {key_id}")
                return False

            # Simplified verification - in production use proper cryptographic verification
            expected = hmac.new(
                self._trusted_keys[key_id],
                f"{plugin_name}:{content_hash}".encode(),
                hashlib.sha256
            ).hexdigest()

            return hmac.compare_digest(sig_data, expected)
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False


# =============================================================================
# Main Safety Gate
# =============================================================================

class PluginSafetyGate:
    """
    The FIRST LINE OF DEFENSE for the Dark Trace plugin ecosystem.

    Every plugin operation must pass through this gate. It enforces:
    - Trust level validation
    - Capability enforcement
    - Risk assessment
    - Blocked plugin detection
    - Signature verification

    Integration:
        Works with HoloLoom's SafetyGuardrails for additional runtime checks.

    Usage:
        gate = PluginSafetyGate(guardrails=guardrails)

        # Validate plugin before registration
        result = await gate.validate_plugin(plugin)
        if not result.allowed:
            logger.error(f"Plugin blocked: {result.reason}")
            return

        # Gate individual operations
        if await gate.gate_operation(plugin_name, PluginCapability.STEER_MODEL, context):
            # Proceed with steering
            pass
    """

    def __init__(
        self,
        guardrails: Any | None = None,
        signature_verifier: SignatureVerifier | None = None,
        blocked_registry: BlockedPluginsRegistry | None = None,
        risk_threshold: float = 0.8,
    ) -> None:
        """
        Initialize the safety gate.

        Args:
            guardrails: HoloLoom SafetyGuardrails instance for additional checks
            signature_verifier: Verifier for plugin signatures
            blocked_registry: Registry of blocked plugins
            risk_threshold: Maximum allowed risk score (0.0-1.0)
        """
        self._guardrails = guardrails
        self._signature_verifier = signature_verifier or SignatureVerifier()
        self._blocked_registry = blocked_registry or BlockedPluginsRegistry()
        self._risk_threshold = risk_threshold

        # Cache trust levels for registered plugins
        self._trust_cache: dict[str, TrustLevel] = {}

        # Operation counters for monitoring
        self._operation_counts: dict[str, int] = {}
        self._blocked_counts: dict[str, int] = {}

        logger.info("PluginSafetyGate initialized")

    # -------------------------------------------------------------------------
    # Blocked Plugin Checks
    # -------------------------------------------------------------------------

    async def is_plugin_blocked(self, plugin_name: str) -> bool:
        """
        Check if a plugin is blocked by name.

        Args:
            plugin_name: Name of the plugin to check

        Returns:
            True if the plugin is blocked, False otherwise
        """
        blocked = self._blocked_registry.is_blocked(plugin_name)
        return blocked is not None

    # -------------------------------------------------------------------------
    # Plugin Validation
    # -------------------------------------------------------------------------

    async def validate_plugin(
        self,
        plugin: "DarkTracePlugin"
    ) -> SafetyCheckResult:
        """
        Validate a plugin before registration.

        This is the MAIN ENTRY POINT for plugin validation. It performs:
        1. Blocked list check
        2. Metadata validation
        3. Trust level assignment
        4. Capability risk assessment
        5. SafetyGuardrails integration check

        Args:
            plugin: The plugin to validate

        Returns:
            SafetyCheckResult with validation outcome
        """
        metadata = plugin.metadata
        warnings: list[str] = []

        # 1. Check blocked list
        blocked = self._blocked_registry.is_blocked(metadata.name)
        if blocked:
            return SafetyCheckResult(
                allowed=False,
                trust_level=TrustLevel.SANDBOXED,
                blocked_capabilities=list(PluginCapability),
                risk_score=1.0,
                reason=f"Plugin is blocked: {blocked.reason}",
                metadata={"blocked_info": {
                    "blocked_at": blocked.blocked_at.isoformat(),
                    "blocked_by": blocked.blocked_by,
                    "severity": blocked.severity
                }}
            )

        # 2. Validate metadata
        metadata_issues = self._validate_metadata(metadata)
        if metadata_issues:
            return SafetyCheckResult(
                allowed=False,
                trust_level=TrustLevel.SANDBOXED,
                blocked_capabilities=[],
                risk_score=0.5,
                reason=f"Invalid metadata: {', '.join(metadata_issues)}",
            )

        # 3. Determine trust level
        trust_level = await self._determine_trust_level(plugin)

        # 4. Calculate risk and determine blocked capabilities
        allowed_caps = TRUST_CAPABILITIES[trust_level]
        requested_caps = self._parse_requested_capabilities(metadata)
        blocked_caps = [c for c in requested_caps if c not in allowed_caps]
        risk_score = self._calculate_risk_score(requested_caps)

        # 5. Check risk threshold
        if risk_score > self._risk_threshold:
            warnings.append(
                f"High risk score ({risk_score:.2f}) - some capabilities may be limited"
            )

        # 6. Optional SafetyGuardrails check
        if self._guardrails is not None:
            guardrails_check = await self._check_guardrails(metadata, trust_level)
            if not guardrails_check["allowed"]:
                return SafetyCheckResult(
                    allowed=False,
                    trust_level=trust_level,
                    blocked_capabilities=blocked_caps,
                    risk_score=risk_score,
                    reason=guardrails_check.get("reason", "Blocked by SafetyGuardrails"),
                    warnings=warnings,
                )

        # Cache trust level
        self._trust_cache[metadata.name] = trust_level

        return SafetyCheckResult(
            allowed=True,
            trust_level=trust_level,
            blocked_capabilities=blocked_caps,
            risk_score=risk_score,
            reason="Plugin validation passed",
            warnings=warnings,
            metadata={
                "allowed_capabilities": [c.value for c in allowed_caps],
                "requested_capabilities": [c.value for c in requested_caps],
            }
        )

    def _validate_metadata(self, metadata: "PluginMetadata") -> list[str]:
        """Validate plugin metadata structure."""
        issues = []

        if not metadata.name or len(metadata.name) < 2:
            issues.append("Invalid plugin name")

        if not metadata.version or not re.match(r'^\d+\.\d+', metadata.version):
            issues.append("Invalid version format")

        if not metadata.author:
            issues.append("Missing author")

        if not metadata.description or len(metadata.description) < 10:
            issues.append("Description too short (min 10 chars)")

        # Check for suspicious patterns in name
        suspicious_patterns = [
            r'(eval|exec|system|import\s*\()',
            r'(__\w+__)',  # Dunder names
            r'(rm\s+-rf|del\s+/)',  # Dangerous commands
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, metadata.name, re.IGNORECASE):
                issues.append(f"Suspicious pattern in name: {pattern}")

        return issues

    async def _determine_trust_level(
        self,
        plugin: "DarkTracePlugin"
    ) -> TrustLevel:
        """Determine appropriate trust level for a plugin."""
        metadata = plugin.metadata

        # Built-in plugins get CORE
        if getattr(plugin, '_is_builtin', False):
            return TrustLevel.CORE

        # Check signature for VERIFIED
        if metadata.signature:
            # Calculate content hash (simplified - would hash actual plugin code)
            content_hash = hashlib.sha256(
                f"{metadata.name}:{metadata.version}:{metadata.author}".encode()
            ).hexdigest()

            if self._signature_verifier.verify_signature(
                metadata.name,
                metadata.signature,
                content_hash
            ):
                return TrustLevel.VERIFIED

        # Default to SANDBOXED
        return TrustLevel.SANDBOXED

    def _parse_requested_capabilities(
        self,
        metadata: "PluginMetadata"
    ) -> list[PluginCapability]:
        """Parse requested capabilities from metadata."""
        caps = []
        for cap_str in metadata.requested_capabilities:
            try:
                if isinstance(cap_str, PluginCapability):
                    caps.append(cap_str)
                else:
                    caps.append(PluginCapability(cap_str))
            except ValueError:
                logger.warning(f"Unknown capability: {cap_str}")
        return caps

    def _calculate_risk_score(
        self,
        capabilities: list[PluginCapability]
    ) -> float:
        """Calculate overall risk score from requested capabilities."""
        if not capabilities:
            return 0.0

        # Weighted average with max boost for high-risk capabilities
        total_weight = 0.0
        for cap in capabilities:
            weight = CAPABILITY_RISK_WEIGHTS.get(cap, 0.5)
            total_weight += weight

        base_score = total_weight / len(capabilities)

        # Boost for high-risk combinations
        high_risk = {
            PluginCapability.STEER_MODEL,
            PluginCapability.MODIFY_ACTIVATIONS,
            PluginCapability.EXECUTE_CODE
        }
        if len(set(capabilities) & high_risk) >= 2:
            base_score = min(1.0, base_score * 1.3)

        return min(1.0, base_score)

    async def _check_guardrails(
        self,
        metadata: "PluginMetadata",
        trust_level: TrustLevel
    ) -> dict[str, Any]:
        """Additional check via HoloLoom SafetyGuardrails."""
        if self._guardrails is None:
            return {"allowed": True}

        try:
            # Create an action request for the guardrails
            from hololoom.alignment import ActionCategory, ActionRequest

            request = ActionRequest(
                action=f"register_plugin:{metadata.name}",
                category=ActionCategory.EXECUTION,
                parameters={
                    "plugin_name": metadata.name,
                    "plugin_version": metadata.version,
                    "trust_level": trust_level.value,
                    "capabilities": [c for c in metadata.requested_capabilities],
                },
                context={"source": "dark_trace_plugins"}
            )

            decision = await self._guardrails.evaluate(request)
            return {
                "allowed": decision.allowed,
                "reason": decision.reason if not decision.allowed else None
            }
        except Exception as e:
            logger.warning(f"SafetyGuardrails check failed: {e}")
            # Fail open with warning (don't block on integration errors)
            return {"allowed": True, "warning": str(e)}

    # -------------------------------------------------------------------------
    # Operation Gating
    # -------------------------------------------------------------------------

    async def gate_operation(
        self,
        plugin_name: str,
        capability: PluginCapability,
        context: dict[str, Any] | None = None
    ) -> OperationGateResult:
        """
        Gate a specific plugin operation.

        Every operation must pass through this gate. It checks:
        1. Plugin trust level allows capability
        2. Optional SafetyGuardrails runtime check

        Args:
            plugin_name: Name of the requesting plugin
            capability: Capability being requested
            context: Additional context for the operation

        Returns:
            OperationGateResult with gating outcome
        """
        context = context or {}

        # Get trust level (default to SANDBOXED if unknown)
        trust_level = self._trust_cache.get(plugin_name, TrustLevel.SANDBOXED)
        allowed_caps = TRUST_CAPABILITIES[trust_level]

        # Track operation count
        key = f"{plugin_name}:{capability.value}"
        self._operation_counts[key] = self._operation_counts.get(key, 0) + 1

        # Check capability
        if capability not in allowed_caps:
            self._blocked_counts[key] = self._blocked_counts.get(key, 0) + 1
            logger.warning(
                f"Blocked operation: {plugin_name} requested {capability.value} "
                f"(trust level: {trust_level.value})"
            )
            return OperationGateResult(
                allowed=False,
                capability=capability,
                plugin_name=plugin_name,
                reason=f"Capability {capability.value} not allowed for trust level {trust_level.value}"
            )

        # Optional runtime check via guardrails
        if self._guardrails is not None:
            try:
                from hololoom.alignment import ActionCategory, ActionRequest

                # Map capability to action category
                category_map = {
                    PluginCapability.READ_ACTIVATIONS: ActionCategory.QUERY,
                    PluginCapability.READ_FEATURES: ActionCategory.QUERY,
                    PluginCapability.WRITE_FEATURES: ActionCategory.MODIFICATION,
                    PluginCapability.STEER_MODEL: ActionCategory.EXECUTION,
                    PluginCapability.MODIFY_ACTIVATIONS: ActionCategory.EXECUTION,
                    PluginCapability.EXTERNAL_NETWORK: ActionCategory.EXTERNAL,
                    PluginCapability.FILE_SYSTEM: ActionCategory.SYSTEM,
                    PluginCapability.EXECUTE_CODE: ActionCategory.EXECUTION,
                }
                category = category_map.get(capability, ActionCategory.EXECUTION)

                request = ActionRequest(
                    action=f"plugin_operation:{capability.value}",
                    category=category,
                    parameters={
                        "plugin_name": plugin_name,
                        "capability": capability.value,
                        **context
                    },
                    context={"source": "dark_trace_plugins"}
                )

                decision = await self._guardrails.evaluate(request)
                if not decision.allowed:
                    self._blocked_counts[key] = self._blocked_counts.get(key, 0) + 1
                    return OperationGateResult(
                        allowed=False,
                        capability=capability,
                        plugin_name=plugin_name,
                        reason=f"Blocked by SafetyGuardrails: {decision.reason}"
                    )
            except Exception as e:
                logger.warning(f"Guardrails check failed: {e}")
                # Continue without guardrails check

        return OperationGateResult(
            allowed=True,
            capability=capability,
            plugin_name=plugin_name,
        )

    # -------------------------------------------------------------------------
    # Trust Level Management
    # -------------------------------------------------------------------------

    def get_trust_level(self, plugin_name: str) -> TrustLevel:
        """Get current trust level for a plugin."""
        return self._trust_cache.get(plugin_name, TrustLevel.SANDBOXED)

    def set_trust_level(self, plugin_name: str, level: TrustLevel) -> None:
        """
        Manually set trust level for a plugin.

        WARNING: This bypasses normal verification. Use with caution.
        """
        logger.warning(f"Manual trust level change: {plugin_name} → {level.value}")
        self._trust_cache[plugin_name] = level

    def promote_to_trusted(self, plugin_name: str, authorized_by: str) -> bool:
        """
        Promote a VERIFIED plugin to TRUSTED.

        Requires explicit authorization.

        Args:
            plugin_name: Name of the plugin to promote
            authorized_by: Who authorized the promotion

        Returns:
            True if promotion succeeded
        """
        current = self._trust_cache.get(plugin_name)
        if current != TrustLevel.VERIFIED:
            logger.error(f"Cannot promote {plugin_name}: not VERIFIED")
            return False

        logger.info(f"Promoting {plugin_name} to TRUSTED (authorized by: {authorized_by})")
        self._trust_cache[plugin_name] = TrustLevel.TRUSTED
        return True

    # -------------------------------------------------------------------------
    # Monitoring
    # -------------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Get gate statistics for monitoring."""
        return {
            "registered_plugins": len(self._trust_cache),
            "trust_levels": {
                level.value: sum(1 for t in self._trust_cache.values() if t == level)
                for level in TrustLevel
            },
            "operation_counts": dict(self._operation_counts),
            "blocked_counts": dict(self._blocked_counts),
            "blocked_plugins": len(self._blocked_registry.list_blocked()),
        }

    def clear_cache(self) -> None:
        """Clear all cached trust levels. Use with caution."""
        logger.warning("Clearing trust level cache")
        self._trust_cache.clear()


# =============================================================================
# Factory Function
# =============================================================================

def create_safety_gate(
    guardrails: Any | None = None,
    risk_threshold: float = 0.8,
    trusted_keys: dict[str, bytes] | None = None
) -> PluginSafetyGate:
    """
    Create a configured PluginSafetyGate.

    Args:
        guardrails: Optional SafetyGuardrails instance
        risk_threshold: Maximum allowed risk score
        trusted_keys: Dict of trusted signing keys

    Returns:
        Configured PluginSafetyGate instance
    """
    verifier = SignatureVerifier(trusted_keys) if trusted_keys else None
    return PluginSafetyGate(
        guardrails=guardrails,
        signature_verifier=verifier,
        risk_threshold=risk_threshold,
    )


# =============================================================================
# Public Exports
# =============================================================================

__all__ = [
    # Core classes
    "PluginSafetyGate",
    "TrustLevel",
    "PluginCapability",
    # Results
    "SafetyCheckResult",
    "OperationGateResult",
    # Blocked plugins
    "BlockedPlugin",
    "BlockedPluginsRegistry",
    # Signature verification
    "SignatureVerifier",
    # Constants
    "TRUST_CAPABILITIES",
    "CAPABILITY_RISK_WEIGHTS",
    # Factory
    "create_safety_gate",
]
