"""
Dark Trace Plugin Alignment Bridge
==================================

Bridges the plugin system to HoloLoom's comprehensive alignment framework.

Core Principle: "No operation goes unaudited"

This module provides:
- Audit trail logging for all plugin operations
- Deception detection (claimed vs actual behavior)
- Instrumental convergence monitoring (power-seeking detection)
- Real-time safety metrics
- Integration with SafetyGuardrails for runtime checks

Integration Points:
    - SafetyGuardrails: Gates all plugin operations
    - AuditTrail: Logs registration, operations, blocks
    - DeceptionDetector: Validates plugin behavior matches claims
    - InstrumentalConvergenceGuard: Monitors for power-seeking patterns

Usage:
    from hololoom.dark_trace.plugins.alignment_bridge import (
        PluginAlignmentBridge,
        create_alignment_bridge
    )

    bridge = create_alignment_bridge()

    # Audit plugin registration
    await bridge.audit_plugin_registration(plugin, trust_level)

    # Audit operations
    await bridge.audit_plugin_operation(
        plugin_name="my_plugin",
        operation="analyze",
        result=result,
        context={"input_dim": 384}
    )

    # Check for deception
    is_deceptive = await bridge.check_for_deception(plugin, result)

Date: 2025-12-22
Phase: 11.2 - Alignment Bridge (Safety Integration)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.dark_trace.plugins.interface import DarkTracePlugin
    from hololoom.dark_trace.plugins.safety_gate import TrustLevel, PluginCapability
    from hololoom.dark_trace.result import TraceResult

logger = logging.getLogger("hololoom.dark_trace.plugins.alignment_bridge")


# =============================================================================
# Audit Event Types
# =============================================================================

class PluginAuditEventType:
    """Event types for plugin audit trail."""
    # Registration events
    PLUGIN_REGISTERED = "PLUGIN_REGISTERED"
    PLUGIN_UNREGISTERED = "PLUGIN_UNREGISTERED"
    PLUGIN_BLOCKED = "PLUGIN_BLOCKED"
    PLUGIN_PROMOTED = "PLUGIN_PROMOTED"

    # Operation events
    PLUGIN_OPERATION = "PLUGIN_OPERATION"
    PLUGIN_OPERATION_BLOCKED = "PLUGIN_OPERATION_BLOCKED"
    PLUGIN_OPERATION_FAILED = "PLUGIN_OPERATION_FAILED"

    # Safety events
    DECEPTION_DETECTED = "DECEPTION_DETECTED"
    POWER_SEEKING_DETECTED = "POWER_SEEKING_DETECTED"
    CAPABILITY_ESCALATION = "CAPABILITY_ESCALATION"

    # System events
    SAFETY_GATE_INITIALIZED = "SAFETY_GATE_INITIALIZED"
    TRUST_LEVEL_CHANGED = "TRUST_LEVEL_CHANGED"


# =============================================================================
# Audit Entry
# =============================================================================

@dataclass
class PluginAuditEntry:
    """A single audit log entry for plugin operations."""
    event_type: str
    plugin_name: Optional[str]
    timestamp: datetime
    data: Dict[str, Any]
    severity: str = "info"  # debug, info, warning, error, critical
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_type": self.event_type,
            "plugin_name": self.plugin_name,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "severity": self.severity,
            "trace_id": self.trace_id,
        }


# =============================================================================
# Deception Detection Results
# =============================================================================

@dataclass
class DeceptionCheckResult:
    """Result of checking for deceptive plugin behavior."""
    is_deceptive: bool
    confidence: float  # 0.0 = definitely not, 1.0 = definitely deceptive
    claimed_behavior: str
    observed_behavior: str
    discrepancies: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "is_deceptive": self.is_deceptive,
            "confidence": self.confidence,
            "claimed_behavior": self.claimed_behavior,
            "observed_behavior": self.observed_behavior,
            "discrepancies": self.discrepancies,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Power-Seeking Detection Results
# =============================================================================

@dataclass
class PowerSeekingCheckResult:
    """Result of checking for power-seeking behavior."""
    is_power_seeking: bool
    confidence: float
    patterns_detected: List[str]
    capabilities_requested: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "is_power_seeking": self.is_power_seeking,
            "confidence": self.confidence,
            "patterns_detected": self.patterns_detected,
            "capabilities_requested": self.capabilities_requested,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Main Alignment Bridge
# =============================================================================

class PluginAlignmentBridge:
    """
    Bridge between Dark Trace plugins and HoloLoom alignment framework.

    Provides comprehensive safety integration:
    - Audit trail logging for complete provenance
    - Deception detection for behavior validation
    - Power-seeking monitoring for capability escalation
    - Real-time safety metrics

    This is the SECOND LAYER OF DEFENSE after SafetyGate.
    """

    def __init__(
        self,
        guardrails: Optional[Any] = None,
        audit_trail: Optional[Any] = None,
        deception_detector: Optional[Any] = None,
        convergence_guard: Optional[Any] = None,
        enable_local_audit: bool = True,
    ) -> None:
        """
        Initialize the alignment bridge.

        Args:
            guardrails: HoloLoom SafetyGuardrails instance
            audit_trail: HoloLoom AuditTrail instance
            deception_detector: HoloLoom DeceptionDetector instance
            convergence_guard: HoloLoom InstrumentalConvergenceGuard instance
            enable_local_audit: Keep local audit log even without AuditTrail
        """
        self._guardrails = guardrails
        self._audit_trail = audit_trail
        self._deception_detector = deception_detector
        self._convergence_guard = convergence_guard
        self._enable_local_audit = enable_local_audit

        # Local audit log (backup if no AuditTrail)
        self._local_audit: List[PluginAuditEntry] = []
        self._max_local_entries = 10000

        # Metrics
        self._metrics = {
            "total_audited": 0,
            "deception_checks": 0,
            "deception_detected": 0,
            "power_seeking_checks": 0,
            "power_seeking_detected": 0,
            "operations_blocked": 0,
        }

        # Plugin behavior tracking for deception detection
        self._plugin_behaviors: Dict[str, List[Dict[str, Any]]] = {}

        logger.info("PluginAlignmentBridge initialized")

    # -------------------------------------------------------------------------
    # Audit Trail
    # -------------------------------------------------------------------------

    async def audit_plugin_registration(
        self,
        plugin: "DarkTracePlugin",
        trust_level: "TrustLevel",
        success: bool = True,
        failure_reason: Optional[str] = None,
        granted_capabilities: Optional[List[Any]] = None,
        denied_capabilities: Optional[List[Any]] = None,
    ) -> None:
        """
        Log plugin registration to audit trail.

        Args:
            plugin: The plugin being registered
            trust_level: Assigned trust level
            success: Whether registration succeeded
            failure_reason: Reason if failed
            granted_capabilities: Capabilities granted to the plugin
            denied_capabilities: Capabilities denied to the plugin
        """
        metadata = plugin.metadata
        event_type = (
            PluginAuditEventType.PLUGIN_REGISTERED
            if success
            else PluginAuditEventType.PLUGIN_BLOCKED
        )

        data = {
            "plugin_name": metadata.name,
            "plugin_version": metadata.version,
            "plugin_author": metadata.author,
            "plugin_type": metadata.plugin_type.value if hasattr(metadata.plugin_type, 'value') else str(metadata.plugin_type),
            "trust_level": trust_level.value if hasattr(trust_level, 'value') else str(trust_level),
            "requested_capabilities": [
                c.value if hasattr(c, 'value') else str(c)
                for c in metadata.requested_capabilities
            ],
            "success": success,
        }

        if failure_reason:
            data["failure_reason"] = failure_reason

        if granted_capabilities:
            data["granted_capabilities"] = [
                c.value if hasattr(c, 'value') else str(c)
                for c in granted_capabilities
            ]

        if denied_capabilities:
            data["denied_capabilities"] = [
                c.value if hasattr(c, 'value') else str(c)
                for c in denied_capabilities
            ]

        await self._log_event(
            event_type=event_type,
            plugin_name=metadata.name,
            data=data,
            severity="warning" if not success else "info",
        )

    async def audit_plugin_unregistration(
        self,
        plugin_name: str,
        trust_level: Optional["TrustLevel"] = None,
        reason: str = "manual",
    ) -> None:
        """Log plugin unregistration."""
        data = {"reason": reason}
        if trust_level is not None:
            data["trust_level"] = trust_level.value if hasattr(trust_level, 'value') else str(trust_level)

        await self._log_event(
            event_type=PluginAuditEventType.PLUGIN_UNREGISTERED,
            plugin_name=plugin_name,
            data=data,
            severity="info",
        )

    async def audit_trust_change(
        self,
        plugin_name: str,
        old_trust: "TrustLevel",
        new_trust: "TrustLevel",
        reason: str,
    ) -> None:
        """Log trust level change (promotion or demotion)."""
        old_val = old_trust.value if hasattr(old_trust, 'value') else str(old_trust)
        new_val = new_trust.value if hasattr(new_trust, 'value') else str(new_trust)

        await self._log_event(
            event_type=PluginAuditEventType.TRUST_LEVEL_CHANGED,
            plugin_name=plugin_name,
            data={
                "old_trust": old_val,
                "new_trust": new_val,
                "reason": reason,
                "is_promotion": self._compare_trust_levels(old_trust, new_trust) < 0,
            },
            severity="info",
        )

    @staticmethod
    def _compare_trust_levels(a: "TrustLevel", b: "TrustLevel") -> int:
        """Compare trust levels. Returns -1 if a < b, 0 if equal, 1 if a > b."""
        from hololoom.dark_trace.plugins.safety_gate import TrustLevel
        order = [TrustLevel.SANDBOXED, TrustLevel.VERIFIED, TrustLevel.TRUSTED, TrustLevel.CORE]
        try:
            return order.index(a) - order.index(b)
        except (ValueError, TypeError):
            return 0

    async def audit_plugin_operation(
        self,
        plugin_name: str,
        operation: str,
        result: Any,
        context: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """
        Audit a plugin operation.

        Every plugin operation should be audited for:
        - Complete provenance tracking
        - Anomaly detection
        - Behavior analysis

        Args:
            plugin_name: Name of the plugin
            operation: Operation performed
            result: Operation result
            context: Additional context
            success: Whether operation succeeded
            error: Error message if failed
        """
        event_type = (
            PluginAuditEventType.PLUGIN_OPERATION
            if success
            else PluginAuditEventType.PLUGIN_OPERATION_FAILED
        )

        data = {
            "operation": operation,
            "success": success,
            "context": context or {},
            "result_type": type(result).__name__ if result is not None else "None",
        }

        if error:
            data["error"] = error

        await self._log_event(
            event_type=event_type,
            plugin_name=plugin_name,
            data=data,
            severity="error" if not success else "debug",
        )

        # Track behavior for deception detection
        self._track_behavior(plugin_name, operation, result, context)

        self._metrics["total_audited"] += 1

    async def audit_operation_blocked(
        self,
        plugin_name: str,
        operation: str,
        capability: "PluginCapability",
        reason: str,
    ) -> None:
        """Log when an operation is blocked."""
        await self._log_event(
            event_type=PluginAuditEventType.PLUGIN_OPERATION_BLOCKED,
            plugin_name=plugin_name,
            data={
                "operation": operation,
                "capability": capability.value if hasattr(capability, 'value') else str(capability),
                "reason": reason,
            },
            severity="warning",
        )
        self._metrics["operations_blocked"] += 1

    async def _log_event(
        self,
        event_type: str,
        plugin_name: Optional[str],
        data: Dict[str, Any],
        severity: str = "info",
        trace_id: Optional[str] = None,
    ) -> None:
        """Log an event to both local and external audit trail."""
        entry = PluginAuditEntry(
            event_type=event_type,
            plugin_name=plugin_name,
            timestamp=datetime.now(),
            data=data,
            severity=severity,
            trace_id=trace_id,
        )

        # Local audit log
        if self._enable_local_audit:
            self._local_audit.append(entry)
            # Trim if too large
            if len(self._local_audit) > self._max_local_entries:
                self._local_audit = self._local_audit[-self._max_local_entries:]

        # External audit trail
        if self._audit_trail is not None:
            try:
                await self._audit_trail.log(
                    event_type=event_type,
                    data={
                        "plugin_name": plugin_name,
                        "severity": severity,
                        **data,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to log to external audit trail: {e}")

        # Log to standard logger as well
        log_method = getattr(logger, severity, logger.info)
        log_method(f"[{event_type}] {plugin_name}: {data}")

    # -------------------------------------------------------------------------
    # Deception Detection
    # -------------------------------------------------------------------------

    def _track_behavior(
        self,
        plugin_name: str,
        operation: str,
        result: Any,
        context: Optional[Dict[str, Any]],
    ) -> None:
        """Track plugin behavior for deception detection."""
        if plugin_name not in self._plugin_behaviors:
            self._plugin_behaviors[plugin_name] = []

        behavior = {
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "result_type": type(result).__name__ if result is not None else "None",
            "context": context or {},
        }

        self._plugin_behaviors[plugin_name].append(behavior)

        # Keep only recent behaviors
        max_behaviors = 100
        if len(self._plugin_behaviors[plugin_name]) > max_behaviors:
            self._plugin_behaviors[plugin_name] = (
                self._plugin_behaviors[plugin_name][-max_behaviors:]
            )

    async def check_for_deception(
        self,
        plugin: "DarkTracePlugin",
        analysis_result: Optional["TraceResult"] = None,
    ) -> DeceptionCheckResult:
        """
        Check if plugin output shows deceptive patterns.

        Compares claimed behavior (from metadata) with observed behavior
        (from operation history and results).

        Args:
            plugin: Plugin to check
            analysis_result: Optional result from plugin analysis

        Returns:
            DeceptionCheckResult with findings
        """
        self._metrics["deception_checks"] += 1

        # Get claimed behavior from plugin
        claimed_behavior = plugin.describe_behavior()
        metadata = plugin.metadata

        # Get observed behaviors
        behaviors = self._plugin_behaviors.get(metadata.name, [])
        observed_operations = [b["operation"] for b in behaviors]

        # Build observed behavior summary
        observed_behavior = f"Operations: {', '.join(set(observed_operations))}"
        if analysis_result is not None:
            observed_behavior += f"; Result type: {type(analysis_result).__name__}"

        discrepancies: List[str] = []
        confidence = 0.0

        # Check for discrepancies
        # 1. If plugin claims to only read but performs writes
        if "read-only" in claimed_behavior.lower() or "analysis only" in claimed_behavior.lower():
            write_ops = {"write", "modify", "create", "update", "delete", "steer"}
            if any(op.lower() in write_ops for op in observed_operations):
                discrepancies.append("Claims read-only but performed write operations")
                confidence += 0.4

        # 2. If plugin claims specific functionality but does something else
        from hololoom.dark_trace.plugins.safety_gate import PluginCapability
        requested_caps = metadata.requested_capabilities
        if PluginCapability.STEER_MODEL in requested_caps or "steer_model" in requested_caps:
            if "steer" not in " ".join(observed_operations).lower():
                discrepancies.append("Requested steering capability but hasn't used it")
                confidence += 0.2

        # 3. Use external deception detector if available
        if self._deception_detector is not None:
            try:
                external_check = await self._deception_detector.check(
                    claimed_behavior=claimed_behavior,
                    actual_behavior=observed_behavior,
                )
                if hasattr(external_check, 'is_deceptive') and external_check.is_deceptive:
                    discrepancies.extend(getattr(external_check, 'discrepancies', []))
                    confidence = max(confidence, getattr(external_check, 'confidence', 0.5))
            except Exception as e:
                logger.warning(f"External deception check failed: {e}")

        is_deceptive = len(discrepancies) > 0 and confidence > 0.3

        result = DeceptionCheckResult(
            is_deceptive=is_deceptive,
            confidence=confidence,
            claimed_behavior=claimed_behavior,
            observed_behavior=observed_behavior,
            discrepancies=discrepancies,
        )

        if is_deceptive:
            self._metrics["deception_detected"] += 1
            await self._log_event(
                event_type=PluginAuditEventType.DECEPTION_DETECTED,
                plugin_name=metadata.name,
                data=result.to_dict(),
                severity="error",
            )

        return result

    # -------------------------------------------------------------------------
    # Instrumental Convergence (Power-Seeking) Detection
    # -------------------------------------------------------------------------

    async def monitor_power_seeking(
        self,
        plugin_name: str,
        requested_capabilities: List["PluginCapability"],
        current_trust_level: "TrustLevel",
    ) -> PowerSeekingCheckResult:
        """
        Monitor for instrumental convergence (power-seeking) behavior.

        Detects patterns that indicate a plugin is trying to:
        - Acquire more capabilities than needed
        - Escalate trust level
        - Access resources beyond its stated purpose

        Args:
            plugin_name: Name of the plugin
            requested_capabilities: Capabilities being requested
            current_trust_level: Current trust level

        Returns:
            PowerSeekingCheckResult with findings
        """
        self._metrics["power_seeking_checks"] += 1

        patterns_detected: List[str] = []
        confidence = 0.0

        # Get capability strings for analysis
        from hololoom.dark_trace.plugins.safety_gate import PluginCapability, TrustLevel

        cap_strings = [
            c.value if hasattr(c, 'value') else str(c)
            for c in requested_capabilities
        ]

        # Pattern 1: Requesting many capabilities
        if len(requested_capabilities) > 5:
            patterns_detected.append("Requesting unusually many capabilities")
            confidence += 0.2

        # Pattern 2: Requesting high-risk capabilities
        high_risk = {
            PluginCapability.MODIFY_REGISTRY,
            PluginCapability.EXECUTE_CODE,
            PluginCapability.FILE_SYSTEM,
            PluginCapability.EXTERNAL_NETWORK,
        }
        high_risk_requested = [c for c in requested_capabilities if c in high_risk]
        if len(high_risk_requested) >= 2:
            patterns_detected.append(
                f"Requesting multiple high-risk capabilities: {[c.value for c in high_risk_requested]}"
            )
            confidence += 0.3

        # Pattern 3: Trust level mismatch
        if current_trust_level == TrustLevel.SANDBOXED:
            if any(c in high_risk for c in requested_capabilities):
                patterns_detected.append(
                    "SANDBOXED plugin requesting high-risk capabilities"
                )
                confidence += 0.4

        # Pattern 4: Use external convergence guard if available
        if self._convergence_guard is not None:
            try:
                external_check = await self._convergence_guard.check(
                    agent_id=f"plugin:{plugin_name}",
                    actions=cap_strings,
                )
                if hasattr(external_check, 'is_converging') and external_check.is_converging:
                    patterns_detected.extend(getattr(external_check, 'patterns', []))
                    confidence = max(confidence, getattr(external_check, 'confidence', 0.5))
            except Exception as e:
                logger.warning(f"External convergence check failed: {e}")

        is_power_seeking = len(patterns_detected) > 0 and confidence > 0.4

        result = PowerSeekingCheckResult(
            is_power_seeking=is_power_seeking,
            confidence=confidence,
            patterns_detected=patterns_detected,
            capabilities_requested=cap_strings,
        )

        if is_power_seeking:
            self._metrics["power_seeking_detected"] += 1
            await self._log_event(
                event_type=PluginAuditEventType.POWER_SEEKING_DETECTED,
                plugin_name=plugin_name,
                data=result.to_dict(),
                severity="error",
            )

        return result

    # -------------------------------------------------------------------------
    # Safety Metrics
    # -------------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Get alignment bridge metrics for monitoring."""
        return {
            **self._metrics,
            "local_audit_entries": len(self._local_audit),
            "tracked_plugins": len(self._plugin_behaviors),
        }

    def get_recent_audit_entries(
        self,
        limit: int = 100,
        plugin_name: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> List[PluginAuditEntry]:
        """
        Get recent audit entries with optional filtering.

        Args:
            limit: Maximum entries to return
            plugin_name: Filter by plugin name
            event_type: Filter by event type

        Returns:
            List of matching audit entries
        """
        entries = self._local_audit

        if plugin_name:
            entries = [e for e in entries if e.plugin_name == plugin_name]

        if event_type:
            entries = [e for e in entries if e.event_type == event_type]

        return entries[-limit:]

    async def query_audit_trail(
        self,
        event_types: Optional[List[str]] = None,
        plugin_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query the audit trail.

        Uses external AuditTrail if available, falls back to local.

        Args:
            event_types: Filter by event types
            plugin_name: Filter by plugin name
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum results

        Returns:
            List of audit entries as dicts
        """
        if self._audit_trail is not None:
            try:
                return await self._audit_trail.query(
                    event_types=event_types,
                    limit=limit,
                )
            except Exception as e:
                logger.warning(f"External audit query failed: {e}")

        # Fall back to local audit
        entries = self._local_audit

        if event_types:
            entries = [e for e in entries if e.event_type in event_types]

        if plugin_name:
            entries = [e for e in entries if e.plugin_name == plugin_name]

        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]

        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]

        return [e.to_dict() for e in entries[-limit:]]

    # -------------------------------------------------------------------------
    # Properties for external access
    # -------------------------------------------------------------------------

    @property
    def guardrails(self) -> Optional[Any]:
        """Get the SafetyGuardrails instance."""
        return self._guardrails

    @property
    def audit_trail(self) -> Optional[Any]:
        """Get the AuditTrail instance."""
        return self._audit_trail


# =============================================================================
# Factory Function
# =============================================================================

def create_alignment_bridge(
    guardrails: Optional[Any] = None,
    audit_trail: Optional[Any] = None,
    auto_create_components: bool = True,
) -> PluginAlignmentBridge:
    """
    Create a configured PluginAlignmentBridge.

    Args:
        guardrails: Optional SafetyGuardrails instance
        audit_trail: Optional AuditTrail instance
        auto_create_components: Auto-create alignment components if not provided

    Returns:
        Configured PluginAlignmentBridge instance
    """
    deception_detector = None
    convergence_guard = None

    if auto_create_components:
        try:
            from hololoom.alignment import (
                create_guardrails,
                create_audit_trail,
                create_detector,
                create_guard,
            )

            if guardrails is None:
                guardrails = create_guardrails()
            if audit_trail is None:
                audit_trail = create_audit_trail()
            deception_detector = create_detector()
            convergence_guard = create_guard()
        except ImportError:
            logger.warning("hololoom.alignment not available, using minimal bridge")

    return PluginAlignmentBridge(
        guardrails=guardrails,
        audit_trail=audit_trail,
        deception_detector=deception_detector,
        convergence_guard=convergence_guard,
    )


# =============================================================================
# Public Exports
# =============================================================================

__all__ = [
    # Core class
    "PluginAlignmentBridge",
    # Event types
    "PluginAuditEventType",
    # Results
    "PluginAuditEntry",
    "DeceptionCheckResult",
    "PowerSeekingCheckResult",
    # Factory
    "create_alignment_bridge",
]
