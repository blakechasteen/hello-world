"""
Dark Trace Plugin Registry: Trust-Enforced Plugin Management

The registry is the central hub for plugin lifecycle management, with
safety and alignment checks enforced at every operation.

Design Principles:
- Trust levels gate registration (no untrusted plugins in production)
- Every registration/unregistration is audited
- Blocked plugins are never loaded
- Dependencies resolved with capability checks

Trust Level Assignment:
- Unknown plugins start at SANDBOXED
- Signature verification promotes to VERIFIED
- HoloLoom team approval promotes to TRUSTED
- Built-in only can be CORE

Usage:
    from hololoom.dark_trace.plugins import PluginRegistry, TrustLevel

    registry = PluginRegistry(safety_gate, alignment_bridge)

    # Register plugin (goes through safety validation)
    result = await registry.register(my_plugin)

    # Get plugins by type
    lenses = registry.get_by_type(PluginType.LENS)

    # Unregister (audited)
    await registry.unregister("my_plugin")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Type, Any, Callable, Awaitable
from datetime import datetime, timezone
from enum import Enum
import logging
import asyncio
from collections import defaultdict

from hololoom.dark_trace.plugins.safety_gate import (
    TrustLevel,
    PluginCapability,
    PluginSafetyGate,
    SafetyCheckResult,
    TRUST_CAPABILITIES,
)
from hololoom.dark_trace.plugins.alignment_bridge import (
    PluginAlignmentBridge,
    PluginAuditEventType,
)
from hololoom.dark_trace.plugins.interface import (
    DarkTracePlugin,
    PluginType,
    PluginState,
    PluginMetadata,
    LensPlugin,
    ValidatorPlugin,
    MonitorPlugin,
    DomainPlugin,
    SteeringPlugin,
    IntegrationPlugin,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Registration Results
# =============================================================================

class RegistrationStatus(Enum):
    """Status codes for plugin registration attempts."""
    SUCCESS = "success"
    BLOCKED = "blocked"                    # On blocked list
    SAFETY_FAILED = "safety_failed"        # Failed safety gate
    CAPABILITY_DENIED = "capability_denied"  # Requested unavailable capabilities
    DEPENDENCY_MISSING = "dependency_missing"
    ALREADY_REGISTERED = "already_registered"
    VERSION_CONFLICT = "version_conflict"
    INITIALIZATION_FAILED = "initialization_failed"
    TRUST_INSUFFICIENT = "trust_insufficient"  # Trust level too low for requested capabilities


@dataclass
class RegistrationResult:
    """Result of a plugin registration attempt."""
    status: RegistrationStatus
    plugin_name: str
    trust_level: Optional[TrustLevel] = None
    blocked_capabilities: List[PluginCapability] = field(default_factory=list)
    reason: Optional[str] = None
    safety_check: Optional[SafetyCheckResult] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    entry: Optional["PluginEntry"] = None  # The created plugin entry (if successful)

    @property
    def success(self) -> bool:
        return self.status == RegistrationStatus.SUCCESS

    def __str__(self) -> str:
        if self.success:
            return f"✓ {self.plugin_name} registered at {self.trust_level.value} trust"
        else:
            return f"✗ {self.plugin_name}: {self.status.value} - {self.reason}"


@dataclass
class UnregistrationResult:
    """Result of a plugin unregistration attempt."""
    success: bool
    plugin_name: str
    reason: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Plugin Entry (Registry Record)
# =============================================================================

@dataclass
class PluginEntry:
    """Complete record of a registered plugin."""
    plugin: DarkTracePlugin
    trust_level: TrustLevel
    state: PluginState
    registered_at: datetime
    granted_capabilities: Set[PluginCapability]
    denied_capabilities: Set[PluginCapability]
    dependency_plugins: Set[str]  # Names of plugins this depends on
    dependent_plugins: Set[str]   # Names of plugins depending on this
    last_activity: datetime
    operation_count: int = 0
    error_count: int = 0

    @property
    def metadata(self) -> PluginMetadata:
        return self.plugin.metadata

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def is_active(self) -> bool:
        return self.state == PluginState.READY

    def record_operation(self, success: bool = True) -> None:
        """Record an operation on this plugin."""
        self.operation_count += 1
        self.last_activity = datetime.now(timezone.utc)
        if not success:
            self.error_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "version": self.metadata.version,
            "type": self.metadata.plugin_type.value,
            "trust_level": self.trust_level.value,
            "state": self.state.value,
            "registered_at": self.registered_at.isoformat(),
            "granted_capabilities": [c.value for c in self.granted_capabilities],
            "denied_capabilities": [c.value for c in self.denied_capabilities],
            "dependency_plugins": list(self.dependency_plugins),
            "dependent_plugins": list(self.dependent_plugins),
            "operation_count": self.operation_count,
            "error_count": self.error_count,
        }


# =============================================================================
# Registry Events (for observers)
# =============================================================================

class RegistryEventType(Enum):
    """Types of registry events for observers."""
    PLUGIN_REGISTERED = "plugin_registered"
    PLUGIN_UNREGISTERED = "plugin_unregistered"
    PLUGIN_STATE_CHANGED = "plugin_state_changed"
    TRUST_LEVEL_CHANGED = "trust_level_changed"
    REGISTRATION_BLOCKED = "registration_blocked"
    OPERATION_DENIED = "operation_denied"


@dataclass
class RegistryEvent:
    """Event emitted by the registry."""
    event_type: RegistryEventType
    plugin_name: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


RegistryEventHandler = Callable[[RegistryEvent], Awaitable[None]]


# =============================================================================
# Plugin Registry
# =============================================================================

class PluginRegistry:
    """
    Central registry for Dark Trace plugins with safety-first design.

    All operations are gated by safety checks and logged to audit trail.
    Trust levels are enforced at registration and during capability grants.

    Features:
    - Trust-based capability assignment
    - Blocked plugins list (known malicious)
    - Dependency tracking and validation
    - Observer pattern for registry events
    - Thread-safe operations
    """

    def __init__(
        self,
        safety_gate: PluginSafetyGate,
        alignment_bridge: PluginAlignmentBridge,
        min_trust_for_steering: TrustLevel = TrustLevel.TRUSTED,
        allow_untrusted_plugins: bool = False,
    ):
        """
        Initialize the plugin registry.

        Args:
            safety_gate: Gate for validating plugins and operations
            alignment_bridge: Bridge to HoloLoom alignment framework
            min_trust_for_steering: Minimum trust level for steering plugins
            allow_untrusted_plugins: Whether to allow SANDBOXED plugins
        """
        self._safety_gate = safety_gate
        self._alignment_bridge = alignment_bridge
        self._min_trust_for_steering = min_trust_for_steering
        self._allow_untrusted_plugins = allow_untrusted_plugins

        # Plugin storage
        self._plugins: Dict[str, PluginEntry] = {}
        self._by_type: Dict[PluginType, Set[str]] = defaultdict(set)
        self._by_trust: Dict[TrustLevel, Set[str]] = defaultdict(set)

        # Event observers
        self._observers: List[RegistryEventHandler] = []

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "total_registrations": 0,
            "successful_registrations": 0,
            "blocked_registrations": 0,
            "unregistrations": 0,
            "trust_promotions": 0,
            "trust_demotions": 0,
        }

    # =========================================================================
    # Registration
    # =========================================================================

    async def register(
        self,
        plugin: DarkTracePlugin,
        engine: Optional[Any] = None,  # DarkTraceEngine passed during init
        force_trust_level: Optional[TrustLevel] = None,  # For testing/admin
    ) -> RegistrationResult:
        """
        Register a plugin with full safety validation.

        Process:
        1. Check if blocked
        2. Validate through safety gate
        3. Assign trust level (or use forced)
        4. Validate capabilities against trust level
        5. Resolve dependencies
        6. Initialize plugin
        7. Audit the registration

        Args:
            plugin: The plugin to register
            engine: DarkTraceEngine for initialization
            force_trust_level: Override trust level (admin only)

        Returns:
            RegistrationResult with status and details
        """
        async with self._lock:
            self._stats["total_registrations"] += 1
            metadata = plugin.metadata
            plugin_name = metadata.name

            # 1. Check if already registered
            if plugin_name in self._plugins:
                existing = self._plugins[plugin_name]
                if existing.metadata.version == metadata.version:
                    return RegistrationResult(
                        status=RegistrationStatus.ALREADY_REGISTERED,
                        plugin_name=plugin_name,
                        reason=f"Plugin {plugin_name} v{metadata.version} already registered"
                    )
                else:
                    return RegistrationResult(
                        status=RegistrationStatus.VERSION_CONFLICT,
                        plugin_name=plugin_name,
                        reason=f"Version conflict: {existing.metadata.version} vs {metadata.version}"
                    )

            # 2. Check if blocked
            if await self._safety_gate.is_plugin_blocked(plugin_name):
                self._stats["blocked_registrations"] += 1
                await self._emit_event(RegistryEvent(
                    event_type=RegistryEventType.REGISTRATION_BLOCKED,
                    plugin_name=plugin_name,
                    data={"reason": "blocked_list"}
                ))

                # Audit blocked attempt
                await self._alignment_bridge.audit_plugin_registration(
                    plugin=plugin,
                    trust_level=TrustLevel.SANDBOXED,
                    success=False,
                    failure_reason="Plugin is on blocked list"
                )

                return RegistrationResult(
                    status=RegistrationStatus.BLOCKED,
                    plugin_name=plugin_name,
                    reason="Plugin is on the blocked list"
                )

            # 3. Validate through safety gate
            safety_result = await self._safety_gate.validate_plugin(plugin)

            if not safety_result.allowed:
                self._stats["blocked_registrations"] += 1
                await self._emit_event(RegistryEvent(
                    event_type=RegistryEventType.REGISTRATION_BLOCKED,
                    plugin_name=plugin_name,
                    data={"reason": "safety_check_failed", "details": safety_result.reason}
                ))

                await self._alignment_bridge.audit_plugin_registration(
                    plugin=plugin,
                    trust_level=safety_result.trust_level,
                    success=False,
                    failure_reason=safety_result.reason
                )

                return RegistrationResult(
                    status=RegistrationStatus.SAFETY_FAILED,
                    plugin_name=plugin_name,
                    trust_level=safety_result.trust_level,
                    blocked_capabilities=safety_result.blocked_capabilities,
                    reason=safety_result.reason,
                    safety_check=safety_result,
                )

            # 4. Determine trust level
            trust_level = force_trust_level or safety_result.trust_level

            # 5. Check minimum trust requirements
            if not self._allow_untrusted_plugins and trust_level == TrustLevel.SANDBOXED:
                return RegistrationResult(
                    status=RegistrationStatus.TRUST_INSUFFICIENT,
                    plugin_name=plugin_name,
                    trust_level=trust_level,
                    reason="SANDBOXED plugins not allowed in this registry"
                )

            # Check steering plugins require sufficient trust
            if isinstance(plugin, SteeringPlugin):
                if self._compare_trust(trust_level, self._min_trust_for_steering) < 0:
                    return RegistrationResult(
                        status=RegistrationStatus.TRUST_INSUFFICIENT,
                        plugin_name=plugin_name,
                        trust_level=trust_level,
                        reason=f"Steering plugins require {self._min_trust_for_steering.value} trust"
                    )

            # 6. Calculate granted capabilities
            allowed_by_trust = TRUST_CAPABILITIES.get(trust_level, frozenset())
            requested = set(metadata.requested_capabilities)
            granted = requested & allowed_by_trust
            denied = requested - allowed_by_trust

            if denied:
                # Some capabilities denied - check if plugin can still function
                logger.warning(
                    f"Plugin {plugin_name} denied capabilities: {[c.value for c in denied]}"
                )

            # 7. Resolve dependencies
            missing_deps = await self._check_dependencies(metadata.dependencies)
            if missing_deps:
                return RegistrationResult(
                    status=RegistrationStatus.DEPENDENCY_MISSING,
                    plugin_name=plugin_name,
                    trust_level=trust_level,
                    reason=f"Missing dependencies: {missing_deps}"
                )

            # 8. Initialize plugin
            try:
                await plugin.initialize(engine, self._safety_gate)
            except Exception as e:
                logger.error(f"Plugin {plugin_name} initialization failed: {e}")
                await self._alignment_bridge.audit_plugin_registration(
                    plugin=plugin,
                    trust_level=trust_level,
                    success=False,
                    failure_reason=f"Initialization error: {str(e)}"
                )
                return RegistrationResult(
                    status=RegistrationStatus.INITIALIZATION_FAILED,
                    plugin_name=plugin_name,
                    trust_level=trust_level,
                    reason=f"Initialization failed: {str(e)}"
                )

            # 9. Create registry entry
            now = datetime.now(timezone.utc)
            entry = PluginEntry(
                plugin=plugin,
                trust_level=trust_level,
                state=PluginState.READY,
                registered_at=now,
                granted_capabilities=granted,
                denied_capabilities=denied,
                dependency_plugins=set(metadata.dependencies),
                dependent_plugins=set(),
                last_activity=now,
            )

            # 10. Update indexes
            self._plugins[plugin_name] = entry
            self._by_type[metadata.plugin_type].add(plugin_name)
            self._by_trust[trust_level].add(plugin_name)

            # Update dependent tracking for dependencies
            for dep_name in metadata.dependencies:
                if dep_name in self._plugins:
                    self._plugins[dep_name].dependent_plugins.add(plugin_name)

            # 11. Audit successful registration
            await self._alignment_bridge.audit_plugin_registration(
                plugin=plugin,
                trust_level=trust_level,
                success=True,
                granted_capabilities=list(granted),
                denied_capabilities=list(denied),
            )

            # 12. Emit event
            await self._emit_event(RegistryEvent(
                event_type=RegistryEventType.PLUGIN_REGISTERED,
                plugin_name=plugin_name,
                data={
                    "trust_level": trust_level.value,
                    "granted_capabilities": [c.value for c in granted],
                }
            ))

            self._stats["successful_registrations"] += 1
            logger.info(f"Plugin {plugin_name} registered at {trust_level.value} trust")

            return RegistrationResult(
                status=RegistrationStatus.SUCCESS,
                plugin_name=plugin_name,
                trust_level=trust_level,
                blocked_capabilities=list(denied),
                safety_check=safety_result,
                entry=entry,  # Include the created plugin entry
            )

    async def unregister(
        self,
        plugin_name: str,
        force: bool = False,
    ) -> UnregistrationResult:
        """
        Unregister a plugin with dependency checking.

        Args:
            plugin_name: Name of plugin to unregister
            force: Force unregister even if other plugins depend on it

        Returns:
            UnregistrationResult with status
        """
        async with self._lock:
            if plugin_name not in self._plugins:
                return UnregistrationResult(
                    success=False,
                    plugin_name=plugin_name,
                    reason="Plugin not found"
                )

            entry = self._plugins[plugin_name]

            # Check for dependents
            if entry.dependent_plugins and not force:
                return UnregistrationResult(
                    success=False,
                    plugin_name=plugin_name,
                    reason=f"Cannot unregister: {entry.dependent_plugins} depend on this plugin"
                )

            # Shutdown plugin
            try:
                await entry.plugin.shutdown()
            except Exception as e:
                logger.warning(f"Plugin {plugin_name} shutdown error: {e}")

            # Update indexes
            self._by_type[entry.metadata.plugin_type].discard(plugin_name)
            self._by_trust[entry.trust_level].discard(plugin_name)

            # Update dependency tracking
            for dep_name in entry.dependency_plugins:
                if dep_name in self._plugins:
                    self._plugins[dep_name].dependent_plugins.discard(plugin_name)

            # Remove from registry
            del self._plugins[plugin_name]

            # Audit
            await self._alignment_bridge.audit_plugin_unregistration(
                plugin_name=plugin_name,
                trust_level=entry.trust_level,
            )

            # Emit event
            await self._emit_event(RegistryEvent(
                event_type=RegistryEventType.PLUGIN_UNREGISTERED,
                plugin_name=plugin_name,
                data={"trust_level": entry.trust_level.value}
            ))

            self._stats["unregistrations"] += 1
            logger.info(f"Plugin {plugin_name} unregistered")

            return UnregistrationResult(
                success=True,
                plugin_name=plugin_name,
            )

    # =========================================================================
    # Trust Level Management
    # =========================================================================

    async def promote_trust(
        self,
        plugin_name: str,
        new_trust: TrustLevel,
        reason: str,
    ) -> bool:
        """
        Promote a plugin's trust level (requires re-validation).

        Args:
            plugin_name: Plugin to promote
            new_trust: New trust level
            reason: Reason for promotion (for audit)

        Returns:
            True if promotion succeeded
        """
        async with self._lock:
            if plugin_name not in self._plugins:
                return False

            entry = self._plugins[plugin_name]
            old_trust = entry.trust_level

            if self._compare_trust(new_trust, old_trust) <= 0:
                logger.warning(f"Cannot promote {plugin_name}: {new_trust} <= {old_trust}")
                return False

            # Re-validate with safety gate
            safety_result = await self._safety_gate.validate_plugin(entry.plugin)
            if not safety_result.allowed:
                logger.warning(f"Promotion denied: {safety_result.reason}")
                return False

            # Update trust
            self._by_trust[old_trust].discard(plugin_name)
            self._by_trust[new_trust].add(plugin_name)
            entry.trust_level = new_trust

            # Recalculate capabilities
            allowed_by_trust = TRUST_CAPABILITIES.get(new_trust, frozenset())
            requested = set(entry.metadata.requested_capabilities)
            entry.granted_capabilities = requested & allowed_by_trust
            entry.denied_capabilities = requested - allowed_by_trust

            # Audit
            await self._alignment_bridge.audit_trust_change(
                plugin_name=plugin_name,
                old_trust=old_trust,
                new_trust=new_trust,
                reason=reason,
            )

            # Emit event
            await self._emit_event(RegistryEvent(
                event_type=RegistryEventType.TRUST_LEVEL_CHANGED,
                plugin_name=plugin_name,
                data={
                    "old_trust": old_trust.value,
                    "new_trust": new_trust.value,
                    "reason": reason,
                }
            ))

            self._stats["trust_promotions"] += 1
            logger.info(f"Plugin {plugin_name} promoted: {old_trust} → {new_trust}")

            return True

    async def demote_trust(
        self,
        plugin_name: str,
        new_trust: TrustLevel,
        reason: str,
    ) -> bool:
        """
        Demote a plugin's trust level (immediate, no validation needed).

        Args:
            plugin_name: Plugin to demote
            new_trust: New (lower) trust level
            reason: Reason for demotion (for audit)

        Returns:
            True if demotion succeeded
        """
        async with self._lock:
            if plugin_name not in self._plugins:
                return False

            entry = self._plugins[plugin_name]
            old_trust = entry.trust_level

            if self._compare_trust(new_trust, old_trust) >= 0:
                logger.warning(f"Cannot demote {plugin_name}: {new_trust} >= {old_trust}")
                return False

            # Update trust
            self._by_trust[old_trust].discard(plugin_name)
            self._by_trust[new_trust].add(plugin_name)
            entry.trust_level = new_trust

            # Recalculate capabilities (will be reduced)
            allowed_by_trust = TRUST_CAPABILITIES.get(new_trust, frozenset())
            requested = set(entry.metadata.requested_capabilities)
            entry.granted_capabilities = requested & allowed_by_trust
            entry.denied_capabilities = requested - allowed_by_trust

            # Audit
            await self._alignment_bridge.audit_trust_change(
                plugin_name=plugin_name,
                old_trust=old_trust,
                new_trust=new_trust,
                reason=reason,
            )

            # Emit event
            await self._emit_event(RegistryEvent(
                event_type=RegistryEventType.TRUST_LEVEL_CHANGED,
                plugin_name=plugin_name,
                data={
                    "old_trust": old_trust.value,
                    "new_trust": new_trust.value,
                    "reason": reason,
                }
            ))

            self._stats["trust_demotions"] += 1
            logger.warning(f"Plugin {plugin_name} demoted: {old_trust} → {new_trust} - {reason}")

            return True

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get(self, plugin_name: str) -> Optional[PluginEntry]:
        """Get a plugin entry by name."""
        return self._plugins.get(plugin_name)

    def get_plugin(self, plugin_name: str) -> Optional[DarkTracePlugin]:
        """Get the plugin instance by name."""
        entry = self._plugins.get(plugin_name)
        return entry.plugin if entry else None

    def get_by_type(self, plugin_type: PluginType) -> List[DarkTracePlugin]:
        """Get all plugins of a specific type."""
        names = self._by_type.get(plugin_type, set())
        return [self._plugins[n].plugin for n in names if n in self._plugins]

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[DarkTracePlugin]:
        """Get all plugins of a specific type (alias for get_by_type)."""
        return self.get_by_type(plugin_type)

    def get_by_trust(self, trust_level: TrustLevel) -> List[DarkTracePlugin]:
        """Get all plugins at a specific trust level."""
        names = self._by_trust.get(trust_level, set())
        return [self._plugins[n].plugin for n in names if n in self._plugins]

    def get_lenses(self) -> List[LensPlugin]:
        """Get all lens plugins."""
        return [p for p in self.get_by_type(PluginType.LENS) if isinstance(p, LensPlugin)]

    def get_validators(self) -> List[ValidatorPlugin]:
        """Get all validator plugins."""
        return [p for p in self.get_by_type(PluginType.VALIDATOR) if isinstance(p, ValidatorPlugin)]

    def get_monitors(self) -> List[MonitorPlugin]:
        """Get all monitor plugins."""
        return [p for p in self.get_by_type(PluginType.MONITOR) if isinstance(p, MonitorPlugin)]

    def get_active_plugins(self) -> List[DarkTracePlugin]:
        """Get all active plugins."""
        return [e.plugin for e in self._plugins.values() if e.is_active]

    def has_capability(self, plugin_name: str, capability: PluginCapability) -> bool:
        """Check if a plugin has a specific capability."""
        entry = self._plugins.get(plugin_name)
        if not entry:
            return False
        return capability in entry.granted_capabilities

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all plugins with their metadata."""
        return [entry.to_dict() for entry in self._plugins.values()]

    def get_plugin_names(self) -> List[str]:
        """Get list of all registered plugin names."""
        return list(self._plugins.keys())

    def get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific plugin.

        Args:
            name: Plugin name

        Returns:
            Plugin info dict or None if not found
        """
        entry = self._plugins.get(name)
        if entry:
            return entry.to_dict()
        return None

    def get_all_plugin_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all plugins as a dict.

        Returns:
            Dict mapping plugin name -> plugin info
        """
        return {name: entry.to_dict() for name, entry in self._plugins.items()}

    @property
    def plugin_count(self) -> int:
        """Total number of registered plugins."""
        return len(self._plugins)

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            **self._stats,
            "active_plugins": sum(1 for e in self._plugins.values() if e.is_active),
            "total_plugins": len(self._plugins),
            "by_trust": {t.value: len(names) for t, names in self._by_trust.items()},
            "by_type": {t.value: len(names) for t, names in self._by_type.items()},
        }

    # =========================================================================
    # Observer Pattern
    # =========================================================================

    def add_observer(self, handler: RegistryEventHandler) -> None:
        """Add an observer for registry events."""
        self._observers.append(handler)

    def remove_observer(self, handler: RegistryEventHandler) -> None:
        """Remove an observer."""
        if handler in self._observers:
            self._observers.remove(handler)

    async def _emit_event(self, event: RegistryEvent) -> None:
        """Emit an event to all observers."""
        for handler in self._observers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Observer error: {e}")

    # =========================================================================
    # Private Helpers
    # =========================================================================

    async def _check_dependencies(self, dependencies: List[str]) -> List[str]:
        """Check if dependencies are satisfied. Returns missing dependencies."""
        missing = []
        for dep in dependencies:
            if dep not in self._plugins:
                missing.append(dep)
            elif not self._plugins[dep].is_active:
                missing.append(f"{dep} (inactive)")
        return missing

    @staticmethod
    def _compare_trust(a: TrustLevel, b: TrustLevel) -> int:
        """Compare trust levels. Returns -1, 0, or 1."""
        order = [TrustLevel.SANDBOXED, TrustLevel.VERIFIED, TrustLevel.TRUSTED, TrustLevel.CORE]
        return order.index(a) - order.index(b)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def shutdown_all(self) -> None:
        """Shutdown all plugins gracefully."""
        async with self._lock:
            # Shutdown in reverse dependency order
            shutdown_order = self._get_shutdown_order()

            for plugin_name in shutdown_order:
                entry = self._plugins.get(plugin_name)
                if entry:
                    try:
                        await entry.plugin.shutdown()
                        entry.state = PluginState.SHUTDOWN
                    except Exception as e:
                        logger.error(f"Shutdown error for {plugin_name}: {e}")

            logger.info(f"Shutdown {len(shutdown_order)} plugins")

    def _get_shutdown_order(self) -> List[str]:
        """Get plugins in reverse dependency order for shutdown."""
        # Simple topological sort
        visited = set()
        order = []

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            entry = self._plugins.get(name)
            if entry:
                for dep in entry.dependent_plugins:
                    visit(dep)
                order.append(name)

        for name in self._plugins:
            visit(name)

        return order  # Dependencies shutdown last


# =============================================================================
# Factory Function
# =============================================================================

def create_registry(
    safety_gate: Optional[PluginSafetyGate] = None,
    alignment_bridge: Optional[PluginAlignmentBridge] = None,
    **kwargs,
) -> PluginRegistry:
    """
    Create a plugin registry with optional defaults.

    Args:
        safety_gate: Safety gate instance (created if None)
        alignment_bridge: Alignment bridge instance (created if None)
        **kwargs: Additional arguments for PluginRegistry

    Returns:
        Configured PluginRegistry
    """
    if safety_gate is None:
        from hololoom.dark_trace.plugins.safety_gate import create_safety_gate
        safety_gate = create_safety_gate()

    if alignment_bridge is None:
        from hololoom.dark_trace.plugins.alignment_bridge import create_alignment_bridge
        alignment_bridge = create_alignment_bridge()

    return PluginRegistry(
        safety_gate=safety_gate,
        alignment_bridge=alignment_bridge,
        **kwargs,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "RegistrationStatus",
    "RegistryEventType",
    # Data classes
    "RegistrationResult",
    "UnregistrationResult",
    "PluginEntry",
    "RegistryEvent",
    # Types
    "RegistryEventHandler",
    # Main class
    "PluginRegistry",
    # Factory
    "create_registry",
]
