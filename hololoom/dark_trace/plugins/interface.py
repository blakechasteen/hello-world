"""
Dark Trace Plugin Interface
===========================

Base classes and interfaces for Dark Trace plugins with MANDATORY SAFETY CONTRACTS.

Core Principle: "Every plugin must declare its behavior transparently"

This module provides:
- Plugin metadata with required capability declarations
- Base plugin class with safety contracts
- Specialized plugin types (Lens, Validator, Monitor, Domain, Steering)
- Lifecycle management interfaces
- Safety-aware initialization patterns

Safety Contracts:
    1. Every plugin MUST declare requested capabilities
    2. Every plugin MUST implement describe_behavior() for deception detection
    3. Every plugin receives SafetyGate injection during initialization
    4. Steering plugins require TRUSTED or higher trust level

Plugin Types:
    - LensPlugin: Custom analysis lenses
    - ValidatorPlugin: Causal validation methods
    - MonitorPlugin: Safety/feature monitoring
    - DomainPlugin: Domain-specific analysis
    - SteeringPlugin: Model steering (TRUSTED+ only)
    - IntegrationPlugin: External system integration

Usage:
    from hololoom.dark_trace.plugins.interface import (
        DarkTracePlugin,
        PluginMetadata,
        PluginType,
        LensPlugin,
    )

    class MyAnalysisPlugin(LensPlugin):
        @property
        def metadata(self) -> PluginMetadata:
            return PluginMetadata(
                name="my_analysis",
                version="1.0.0",
                author="Me",
                description="Analyzes activations for patterns",
                plugin_type=PluginType.LENS,
                dependencies=[],
                requested_capabilities=[PluginCapability.READ_ACTIVATIONS],
            )

        def describe_behavior(self) -> str:
            return "Reads activations and identifies patterns without modification"

Date: 2025-12-22
Phase: 11.3 - Plugin Interface (Base Classes)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from hololoom.dark_trace.engine import DarkTraceEngine
    from hololoom.dark_trace.plugins.safety_gate import PluginSafetyGate, PluginCapability
    from hololoom.dark_trace.protocol import TraceLens, CausalValidator, Activations, SteeringVector

logger = logging.getLogger("hololoom.dark_trace.plugins.interface")


# =============================================================================
# Plugin Types
# =============================================================================

class PluginType(Enum):
    """
    Types of plugins supported by Dark Trace.

    Each type has specific capabilities and restrictions.
    """
    LENS = "lens"                # Analysis lens (feature extraction)
    VALIDATOR = "validator"      # Causal validator
    MONITOR = "monitor"          # Safety/feature monitoring
    DOMAIN = "domain"            # Domain-specific analysis
    STEERING = "steering"        # Model steering (TRUSTED+ only)
    INTEGRATION = "integration"  # External system integration


class PluginState(Enum):
    """Plugin lifecycle states."""
    CREATED = "created"          # Just instantiated
    VALIDATING = "validating"    # Being validated by SafetyGate
    INITIALIZING = "initializing"  # Being initialized
    READY = "ready"              # Ready for use
    RUNNING = "running"          # Currently executing
    SUSPENDED = "suspended"      # Temporarily suspended
    SHUTTING_DOWN = "shutting_down"  # Being shut down
    TERMINATED = "terminated"    # Shut down complete
    ERROR = "error"              # Error state


# =============================================================================
# Plugin Metadata
# =============================================================================

@dataclass
class PluginMetadata:
    """
    Plugin metadata with MANDATORY safety declarations.

    Every plugin must provide complete metadata including:
    - Identity (name, version, author)
    - Description (for documentation and deception detection)
    - Type (determines allowed operations)
    - Dependencies (other plugins required)
    - Requested capabilities (REQUIRED for safety gating)
    - Optional signature (for trust level promotion)

    Attributes:
        name: Unique plugin identifier
        version: Semantic version (e.g., "1.0.0")
        author: Plugin author name/email
        description: Human-readable description (min 10 chars)
        plugin_type: Type of plugin (determines capabilities)
        dependencies: List of required plugin names
        requested_capabilities: Capabilities the plugin needs
        signature: Optional cryptographic signature for verification
        min_dark_trace_version: Minimum Dark Trace version required
        tags: Optional categorization tags
        homepage: Optional URL to plugin homepage
        license: Optional license identifier
    """
    name: str
    version: str
    author: str
    description: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    requested_capabilities: List[Union["PluginCapability", str]] = field(default_factory=list)
    signature: Optional[str] = None
    min_dark_trace_version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    homepage: Optional[str] = None
    license: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "plugin_type": self.plugin_type.value,
            "dependencies": self.dependencies,
            "requested_capabilities": [
                c.value if hasattr(c, 'value') else str(c)
                for c in self.requested_capabilities
            ],
            "signature": self.signature,
            "min_dark_trace_version": self.min_dark_trace_version,
            "tags": self.tags,
            "homepage": self.homepage,
            "license": self.license,
        }

    def validate(self) -> List[str]:
        """
        Validate metadata completeness.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not self.name or len(self.name) < 2:
            errors.append("Plugin name must be at least 2 characters")

        if not self.version:
            errors.append("Plugin version is required")

        if not self.author:
            errors.append("Plugin author is required")

        if not self.description or len(self.description) < 10:
            errors.append("Plugin description must be at least 10 characters")

        if self.plugin_type is None:
            errors.append("Plugin type is required")

        return errors


# =============================================================================
# Plugin Lifecycle Events
# =============================================================================

@dataclass
class PluginLifecycleEvent:
    """Event emitted during plugin lifecycle."""
    event_type: str  # "initialized", "started", "stopped", "error"
    plugin_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


LifecycleCallback = Callable[[PluginLifecycleEvent], None]


# =============================================================================
# Base Plugin Class
# =============================================================================

class DarkTracePlugin(ABC):
    """
    Base class for all Dark Trace plugins with MANDATORY SAFETY CONTRACTS.

    Every plugin MUST implement:
    1. metadata property - Full plugin metadata including capabilities
    2. describe_behavior() - Human-readable behavior description
    3. initialize() - Initialization with safety gate injection
    4. shutdown() - Clean resource cleanup

    Safety Contracts:
    - Plugins receive SafetyGate during initialization
    - All operations must be checked against the gate
    - Behavior description must match actual behavior (deception detection)
    - Plugins must not exceed their declared capabilities

    Usage:
        class MyPlugin(DarkTracePlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(...)

            def describe_behavior(self) -> str:
                return "Analyzes activations without modification"

            async def initialize(self, engine, safety_gate):
                self._gate = safety_gate
                # Plugin-specific initialization

            async def shutdown(self):
                # Clean up resources
    """

    # Internal state tracking
    _state: PluginState = PluginState.CREATED
    _engine: Optional["DarkTraceEngine"] = None
    _safety_gate: Optional["PluginSafetyGate"] = None
    _lifecycle_callbacks: List[LifecycleCallback] = []

    # Flag for built-in plugins (CORE trust level)
    _is_builtin: bool = False

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """
        Return plugin metadata including safety declarations.

        REQUIRED: This is how the plugin declares its identity,
        capabilities, and dependencies.

        Returns:
            PluginMetadata with complete plugin information
        """
        pass

    @abstractmethod
    def describe_behavior(self) -> str:
        """
        Return human-readable description of what this plugin does.

        REQUIRED: Used for deception detection. The description must
        accurately reflect the plugin's actual behavior. Discrepancies
        between claimed and observed behavior will flag the plugin.

        Returns:
            Clear description of plugin behavior

        Example:
            "Reads activations and identifies safety-relevant features.
             Does not modify activations or perform any steering."
        """
        pass

    @abstractmethod
    async def initialize(
        self,
        engine: "DarkTraceEngine",
        safety_gate: "PluginSafetyGate",
    ) -> None:
        """
        Initialize the plugin with engine and safety gate.

        REQUIRED: Called after safety validation passes. The plugin
        receives the engine for accessing Dark Trace capabilities and
        the safety gate for checking operation permissions.

        Args:
            engine: DarkTraceEngine instance
            safety_gate: PluginSafetyGate for operation checking

        Note:
            Store references to engine and safety_gate for later use.
            Check safety_gate before any sensitive operations.
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Clean shutdown of plugin resources.

        REQUIRED: Called when plugin is being unloaded. Must release
        all resources, close connections, and clean up state.
        """
        pass

    # -------------------------------------------------------------------------
    # State Management (provided by base class)
    # -------------------------------------------------------------------------

    @property
    def state(self) -> PluginState:
        """Get current plugin state."""
        return self._state

    def _set_state(self, state: PluginState) -> None:
        """Set plugin state and emit lifecycle event."""
        old_state = self._state
        self._state = state

        event = PluginLifecycleEvent(
            event_type=f"state_changed:{old_state.value}->{state.value}",
            plugin_name=self.metadata.name,
            details={"old_state": old_state.value, "new_state": state.value}
        )

        for callback in self._lifecycle_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Lifecycle callback error: {e}")

    def add_lifecycle_callback(self, callback: LifecycleCallback) -> None:
        """Add a lifecycle event callback."""
        self._lifecycle_callbacks.append(callback)

    def remove_lifecycle_callback(self, callback: LifecycleCallback) -> None:
        """Remove a lifecycle event callback."""
        if callback in self._lifecycle_callbacks:
            self._lifecycle_callbacks.remove(callback)

    # -------------------------------------------------------------------------
    # Utility Methods (provided by base class)
    # -------------------------------------------------------------------------

    async def check_capability(self, capability: "PluginCapability") -> bool:
        """
        Check if an operation is allowed.

        Convenience method for checking capabilities against the safety gate.

        Args:
            capability: Capability to check

        Returns:
            True if operation is allowed
        """
        if self._safety_gate is None:
            logger.warning(f"{self.metadata.name}: No safety gate, denying operation")
            return False

        result = await self._safety_gate.gate_operation(
            plugin_name=self.metadata.name,
            capability=capability,
            context={}
        )
        return result.allowed

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.metadata.name} state={self._state.value}>"


# =============================================================================
# Specialized Plugin Types
# =============================================================================

class LensPlugin(DarkTracePlugin):
    """
    Plugin providing a custom TraceLens for analysis.

    Lens plugins provide new ways to analyze activations by implementing
    the TraceLens protocol. They can extract features, identify patterns,
    and contribute to the trace result.

    Required Capabilities:
        - READ_ACTIVATIONS (to read input)
        - WRITE_FEATURES (to contribute features)

    Example:
        class SafetyLensPlugin(LensPlugin):
            def get_lens_class(self) -> Type[TraceLens]:
                return MySafetyLens

            def describe_behavior(self) -> str:
                return "Identifies safety-relevant features in activations"
    """

    @abstractmethod
    def get_lens_class(self) -> Type["TraceLens"]:
        """
        Return the TraceLens class implemented by this plugin.

        Returns:
            TraceLens subclass
        """
        pass

    def get_lens_config(self) -> Optional[Dict[str, Any]]:
        """
        Return optional configuration for the lens.

        Override to provide custom configuration.

        Returns:
            Configuration dict or None
        """
        return None


class ValidatorPlugin(DarkTracePlugin):
    """
    Plugin providing a causal validator for feature verification.

    Validator plugins implement causal validation methods like
    ablation, injection, or patching to verify feature importance.

    Required Capabilities:
        - READ_ACTIVATIONS
        - MODIFY_ACTIVATIONS (for causal interventions)

    Example:
        class AblationValidatorPlugin(ValidatorPlugin):
            def get_validator_class(self) -> Type[CausalValidator]:
                return MyAblationValidator
    """

    @abstractmethod
    def get_validator_class(self) -> Type["CausalValidator"]:
        """
        Return the CausalValidator class implemented by this plugin.

        Returns:
            CausalValidator subclass
        """
        pass


class MonitorPlugin(DarkTracePlugin):
    """
    Plugin for monitoring safety features or system behavior.

    Monitor plugins observe system behavior and emit alerts or metrics.
    They typically have read-only access and don't modify state.

    Required Capabilities:
        - READ_ACTIVATIONS
        - READ_FEATURES

    Example:
        class SafetyMonitorPlugin(MonitorPlugin):
            async def on_analysis_complete(self, result):
                if result.has_safety_flags():
                    await self.emit_alert(result)
    """

    @abstractmethod
    async def on_analysis_complete(
        self,
        result: "TraceResult",
    ) -> None:
        """
        Called after each analysis completes.

        Args:
            result: The trace result from analysis
        """
        pass

    async def on_steering_applied(
        self,
        vector: "SteeringVector",
        activations: "Activations",
    ) -> None:
        """
        Called after steering is applied.

        Optional override for steering monitoring.

        Args:
            vector: The steering vector that was applied
            activations: The activations that were steered
        """
        pass


class DomainPlugin(DarkTracePlugin):
    """
    Plugin providing domain-specific analysis configuration.

    Domain plugins configure the engine for specific domains like
    sentiment analysis, code understanding, or medical text.

    Required Capabilities:
        - READ_CONFIG

    Example:
        class MedicalDomainPlugin(DomainPlugin):
            def get_domain_config(self) -> DomainConfig:
                return DomainConfig(
                    name="medical",
                    safety_keywords=["diagnosis", "medication"],
                    steering_limits={"caution": 0.8}
                )
    """

    @abstractmethod
    def get_domain_config(self) -> Any:  # DomainConfig
        """
        Return domain-specific configuration.

        Returns:
            DomainConfig for this domain
        """
        pass


class SteeringPlugin(DarkTracePlugin):
    """
    Plugin with model steering capability.

    IMPORTANT: Requires TRUSTED or higher trust level.

    Steering plugins can modify model behavior by applying steering
    vectors to activations. This is a high-risk capability that
    requires explicit trust verification.

    Required Capabilities:
        - STEER_MODEL
        - READ_ACTIVATIONS
        - MODIFY_ACTIVATIONS

    Example:
        class SafetySteeringPlugin(SteeringPlugin):
            async def steer(self, activations, goals, safety_gate):
                # Verify we can steer
                if not await safety_gate.gate_operation(...):
                    raise PermissionError("Steering not allowed")
                return self._compute_steering_vector(activations, goals)
    """

    @abstractmethod
    async def steer(
        self,
        activations: "Activations",
        goals: Dict[str, float],
        safety_gate: "PluginSafetyGate",
    ) -> "SteeringVector":
        """
        Compute and return steering vector.

        IMPORTANT: Must check safety_gate before applying steering.

        Args:
            activations: Input activations
            goals: Steering goals (e.g., {"safety": 0.8})
            safety_gate: Gate for permission checking

        Returns:
            SteeringVector to apply

        Raises:
            PermissionError: If steering not allowed
        """
        pass


class IntegrationPlugin(DarkTracePlugin):
    """
    Plugin for integrating with external systems.

    Integration plugins connect Dark Trace to external services,
    databases, or APIs. They may require network access.

    Required Capabilities:
        - EXTERNAL_NETWORK (if accessing network)
        - FILE_SYSTEM (if accessing files)

    Example:
        class PrometheusExporterPlugin(IntegrationPlugin):
            async def export_metrics(self, metrics):
                await self._prometheus_client.push(metrics)
    """

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to external system.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from external system."""
        pass


# =============================================================================
# Plugin Context
# =============================================================================

@dataclass
class PluginContext:
    """
    Context passed to plugins during operations.

    Provides access to engine, safety gate, and shared state.
    """
    engine: "DarkTraceEngine"
    safety_gate: "PluginSafetyGate"
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Plugin Hooks
# =============================================================================

class PluginHook:
    """
    Hook points for plugin extension.

    Plugins can register callbacks at these points in the analysis pipeline.
    """
    # Analysis pipeline hooks
    PRE_ANALYSIS = "pre_analysis"       # Before analysis starts
    POST_ANALYSIS = "post_analysis"     # After analysis completes
    PRE_STEERING = "pre_steering"       # Before steering applied
    POST_STEERING = "post_steering"     # After steering applied

    # Registry hooks
    PLUGIN_REGISTERED = "plugin_registered"
    PLUGIN_UNREGISTERED = "plugin_unregistered"

    # Safety hooks
    SAFETY_FLAG_RAISED = "safety_flag_raised"
    DECEPTION_DETECTED = "deception_detected"


HookCallback = Callable[[str, Dict[str, Any]], None]


class HookRegistry:
    """Registry for plugin hooks."""

    def __init__(self) -> None:
        self._hooks: Dict[str, List[HookCallback]] = {}

    def register(self, hook_point: str, callback: HookCallback) -> None:
        """Register a callback at a hook point."""
        if hook_point not in self._hooks:
            self._hooks[hook_point] = []
        self._hooks[hook_point].append(callback)

    def unregister(self, hook_point: str, callback: HookCallback) -> None:
        """Unregister a callback from a hook point."""
        if hook_point in self._hooks:
            if callback in self._hooks[hook_point]:
                self._hooks[hook_point].remove(callback)

    def emit(self, hook_point: str, data: Dict[str, Any]) -> None:
        """Emit event to all registered callbacks."""
        if hook_point in self._hooks:
            for callback in self._hooks[hook_point]:
                try:
                    callback(hook_point, data)
                except Exception as e:
                    logger.error(f"Hook callback error at {hook_point}: {e}")

    def clear(self, hook_point: Optional[str] = None) -> None:
        """Clear hooks (all or specific point)."""
        if hook_point:
            self._hooks.pop(hook_point, None)
        else:
            self._hooks.clear()


# =============================================================================
# Built-in Plugin Marker
# =============================================================================

def builtin_plugin(cls: Type[DarkTracePlugin]) -> Type[DarkTracePlugin]:
    """
    Decorator to mark a plugin as built-in (CORE trust level).

    Built-in plugins are part of Dark Trace and receive full privileges.

    Usage:
        @builtin_plugin
        class SafetyMonitorPlugin(MonitorPlugin):
            ...
    """
    cls._is_builtin = True
    return cls


# =============================================================================
# Public Exports
# =============================================================================

__all__ = [
    # Enums
    "PluginType",
    "PluginState",
    # Metadata
    "PluginMetadata",
    # Base class
    "DarkTracePlugin",
    # Specialized types
    "LensPlugin",
    "ValidatorPlugin",
    "MonitorPlugin",
    "DomainPlugin",
    "SteeringPlugin",
    "IntegrationPlugin",
    # Context
    "PluginContext",
    # Hooks
    "PluginHook",
    "HookCallback",
    "HookRegistry",
    # Lifecycle
    "PluginLifecycleEvent",
    "LifecycleCallback",
    # Decorators
    "builtin_plugin",
]
