# Dark Trace Plugin Development Guide

**Version**: 1.0.0
**Date**: December 2025
**Phase**: 11 (Ecosystem)

Complete guide for developing, testing, and publishing plugins for the Dark Trace interpretability framework.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Plugin Architecture](#plugin-architecture)
3. [Trust Levels & Safety](#trust-levels--safety)
4. [Plugin Types](#plugin-types)
5. [API Reference](#api-reference)
6. [Lifecycle & Hooks](#lifecycle--hooks)
7. [Testing Plugins](#testing-plugins)
8. [Publishing Plugins](#publishing-plugins)

---

## Quick Start

### Your First Plugin in 5 Minutes

Create a minimal analysis plugin:

```python
from HoloLoom.dark_trace.plugins.interface import (
    DarkTracePlugin,
    PluginMetadata,
    PluginType,
)
from HoloLoom.dark_trace.plugins.safety_gate import PluginCapability

class MyFirstPlugin(DarkTracePlugin):
    """A simple plugin that logs activation statistics."""

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my_first_plugin",
            version="1.0.0",
            author="Your Name",
            description="Logs basic statistics about model activations",
            plugin_type=PluginType.MONITOR,
            requested_capabilities=[
                PluginCapability.READ_ACTIVATIONS,
            ],
        )

    def describe_behavior(self) -> str:
        """Required: Describe what your plugin does (for safety verification)."""
        return "Reads activations and logs statistics. Does not modify any data."

    async def initialize(self, engine, safety_gate) -> None:
        """Called when plugin is loaded."""
        self._engine = engine
        self._safety_gate = safety_gate
        print(f"Plugin {self.metadata.name} initialized!")

    async def shutdown(self) -> None:
        """Called when plugin is unloaded."""
        print(f"Plugin {self.metadata.name} shutting down")
```

### Registering Your Plugin

```python
from HoloLoom.dark_trace.plugins.registry import PluginRegistry

# Create registry and register plugin
registry = PluginRegistry()
plugin = MyFirstPlugin()
await registry.register(plugin)

# List registered plugins
print(registry.list_plugins())
```

---

## Plugin Architecture

### Core Concepts

```
┌─────────────────────────────────────────────────────────────────┐
│                     Plugin Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   Plugin Request                                                  │
│        ↓                                                          │
│   ┌─────────────────┐                                            │
│   │  Safety Gate    │ ← Trust Level Validation                   │
│   │  (First Check)  │ ← Capability Verification                  │
│   └────────┬────────┘ ← Risk Assessment                          │
│            ↓                                                      │
│   ┌─────────────────┐                                            │
│   │    Registry     │ ← Plugin Storage                           │
│   │                 │ ← Dependency Resolution                    │
│   └────────┬────────┘ ← Lifecycle Management                     │
│            ↓                                                      │
│   ┌─────────────────┐                                            │
│   │  Plugin Loader  │ ← Dynamic Loading                          │
│   │                 │ ← Hook Registration                        │
│   └────────┬────────┘                                            │
│            ↓                                                      │
│   ┌─────────────────┐                                            │
│   │  Dark Trace     │ ← Engine Integration                       │
│   │    Engine       │ ← Activation Analysis                      │
│   └─────────────────┘                                            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| **Safety Gate** | `safety_gate.py` | Trust validation, capability checking |
| **Registry** | `registry.py` | Plugin storage, lookup, dependency resolution |
| **Interface** | `interface.py` | Base classes, type definitions |
| **Protocol** | `plugin_protocol.py` | Data structures, category enums |
| **Loader** | `plugin_loader.py` | Dynamic loading, lifecycle management |
| **Discovery** | `discovery.py` | Entry point discovery |

---

## Trust Levels & Safety

### Trust Level Hierarchy

Dark Trace uses a 4-tier trust model inspired by browser security:

| Trust Level | Description | Use Case |
|-------------|-------------|----------|
| **SANDBOXED** | Default for unknown plugins. Read-only access. | Third-party plugins |
| **VERIFIED** | Code reviewed and signed. Can write features. | Community plugins |
| **TRUSTED** | HoloLoom team approved. Full non-system access. | Official plugins |
| **CORE** | Built-in only. System-level access. | Built-in plugins |

### Capabilities

Each trust level grants specific capabilities:

```python
from HoloLoom.dark_trace.plugins.safety_gate import PluginCapability, TrustLevel

# SANDBOXED capabilities (read-only)
SANDBOXED_CAPS = {
    PluginCapability.READ_ACTIVATIONS,    # Read model activations
    PluginCapability.READ_FEATURES,        # Read feature registry
    PluginCapability.READ_CONFIG,          # Read engine configuration
}

# VERIFIED adds writing capabilities
VERIFIED_CAPS = SANDBOXED_CAPS | {
    PluginCapability.WRITE_FEATURES,       # Write to feature registry
    PluginCapability.REGISTER_LENS,        # Register custom lens
    PluginCapability.REGISTER_VALIDATOR,   # Register causal validator
}

# TRUSTED adds steering capabilities
TRUSTED_CAPS = VERIFIED_CAPS | {
    PluginCapability.STEER_MODEL,          # Apply steering vectors
    PluginCapability.MODIFY_ACTIVATIONS,   # Modify live activations
    PluginCapability.EXTERNAL_NETWORK,     # Make network requests
}

# CORE has all capabilities (built-in only)
CORE_CAPS = set(PluginCapability)  # All 12 capabilities
```

### Requesting Capabilities

Declare needed capabilities in your plugin metadata:

```python
@property
def metadata(self) -> PluginMetadata:
    return PluginMetadata(
        name="my_plugin",
        version="1.0.0",
        author="You",
        description="Description of at least 10 characters",
        plugin_type=PluginType.MONITOR,
        requested_capabilities=[
            PluginCapability.READ_ACTIVATIONS,
            PluginCapability.READ_FEATURES,
        ],
    )
```

### Safety Validation

Before registration, plugins are validated by the Safety Gate:

```python
from HoloLoom.dark_trace.plugins.safety_gate import PluginSafetyGate

gate = PluginSafetyGate()
result = await gate.validate_plugin(plugin)

if result.allowed:
    print(f"Plugin approved with trust level: {result.trust_level}")
else:
    print(f"Plugin rejected: {result.reason}")
```

---

## Plugin Types

### 6 Specialized Plugin Types

| Type | Base Class | Purpose | Required Capabilities |
|------|------------|---------|----------------------|
| **Lens** | `LensPlugin` | Custom feature extraction | READ_ACTIVATIONS, WRITE_FEATURES |
| **Validator** | `ValidatorPlugin` | Causal validation methods | READ_ACTIVATIONS, MODIFY_ACTIVATIONS |
| **Monitor** | `MonitorPlugin` | Safety/feature monitoring | READ_ACTIVATIONS, READ_FEATURES |
| **Domain** | `DomainPlugin` | Domain-specific configuration | READ_CONFIG |
| **Steering** | `SteeringPlugin` | Model steering (TRUSTED+ only) | STEER_MODEL, MODIFY_ACTIVATIONS |
| **Integration** | `IntegrationPlugin` | External system integration | EXTERNAL_NETWORK |

### Example: Lens Plugin

```python
from HoloLoom.dark_trace.plugins.interface import LensPlugin, PluginMetadata, PluginType
from HoloLoom.dark_trace.protocol import TraceLens

class MySafetyLens(TraceLens):
    """Custom lens that identifies safety-relevant features."""

    def extract_features(self, activations):
        # Your feature extraction logic
        return {"safety_score": 0.85}

class SafetyLensPlugin(LensPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="safety_lens",
            version="1.0.0",
            author="HoloLoom Team",
            description="Identifies safety-relevant features in activations",
            plugin_type=PluginType.LENS,
            requested_capabilities=[
                PluginCapability.READ_ACTIVATIONS,
                PluginCapability.WRITE_FEATURES,
            ],
        )

    def describe_behavior(self) -> str:
        return "Analyzes activations for safety-relevant patterns. Read-only analysis."

    def get_lens_class(self):
        return MySafetyLens
```

### Example: Monitor Plugin

```python
from HoloLoom.dark_trace.plugins.interface import MonitorPlugin, PluginMetadata, PluginType

class AlertMonitorPlugin(MonitorPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="alert_monitor",
            version="1.0.0",
            author="You",
            description="Monitors analysis results and sends alerts on safety flags",
            plugin_type=PluginType.MONITOR,
            requested_capabilities=[
                PluginCapability.READ_ACTIVATIONS,
                PluginCapability.READ_FEATURES,
            ],
        )

    def describe_behavior(self) -> str:
        return "Observes analysis results and logs alerts. Does not modify state."

    async def on_analysis_complete(self, result):
        if hasattr(result, 'safety_flags') and result.safety_flags:
            print(f"ALERT: Safety flags detected: {result.safety_flags}")
```

### Example: Steering Plugin (TRUSTED+ only)

```python
from HoloLoom.dark_trace.plugins.interface import SteeringPlugin, PluginMetadata, PluginType

class SafetySteeringPlugin(SteeringPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="safety_steering",
            version="1.0.0",
            author="HoloLoom Team",
            description="Applies safety-focused steering vectors to model outputs",
            plugin_type=PluginType.STEERING,
            requested_capabilities=[
                PluginCapability.STEER_MODEL,
                PluginCapability.MODIFY_ACTIVATIONS,
                PluginCapability.READ_ACTIVATIONS,
            ],
        )

    def describe_behavior(self) -> str:
        return "Computes and applies steering vectors to increase safety alignment."

    async def steer(self, activations, goals, safety_gate):
        # Always check safety gate before steering
        result = await safety_gate.gate_operation(
            plugin_name=self.metadata.name,
            capability=PluginCapability.STEER_MODEL,
            context={"goals": goals}
        )
        if not result.allowed:
            raise PermissionError(f"Steering not allowed: {result.reason}")

        # Compute steering vector
        return self._compute_safety_steering(activations, goals)
```

---

## API Reference

### PluginMetadata

```python
@dataclass
class PluginMetadata:
    name: str                        # Unique identifier (2+ chars)
    version: str                     # Semantic version (e.g., "1.0.0")
    author: str                      # Author name/email
    description: str                 # Description (10+ chars)
    plugin_type: PluginType          # LENS, VALIDATOR, MONITOR, etc.
    dependencies: List[str] = []     # Required plugin names
    requested_capabilities: List[PluginCapability] = []  # Needed capabilities
    signature: Optional[str] = None  # Cryptographic signature (for VERIFIED+)
    min_dark_trace_version: str = "1.0.0"
    tags: List[str] = []            # Discovery tags
    homepage: Optional[str] = None
    license: Optional[str] = None
```

### DarkTracePlugin Base Class

```python
class DarkTracePlugin(ABC):
    # Required properties
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass

    # Required methods
    @abstractmethod
    def describe_behavior(self) -> str:
        """Return human-readable description for safety verification."""
        pass

    @abstractmethod
    async def initialize(self, engine, safety_gate) -> None:
        """Initialize plugin with engine and safety gate."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean shutdown and resource release."""
        pass

    # Provided methods
    @property
    def state(self) -> PluginState:
        """Get current plugin state."""
        pass

    async def check_capability(self, capability) -> bool:
        """Check if operation is allowed."""
        pass
```

### PluginRegistry

```python
class PluginRegistry:
    async def register(self, plugin: DarkTracePlugin) -> bool:
        """Register a plugin after safety validation."""
        pass

    async def unregister(self, name: str) -> bool:
        """Unregister a plugin."""
        pass

    def get(self, name: str) -> Optional[DarkTracePlugin]:
        """Get plugin by name."""
        pass

    def list_plugins(self, plugin_type: Optional[PluginType] = None) -> List[str]:
        """List registered plugins, optionally filtered by type."""
        pass

    def get_by_type(self, plugin_type: PluginType) -> List[DarkTracePlugin]:
        """Get all plugins of a specific type."""
        pass
```

### PluginSafetyGate

```python
class PluginSafetyGate:
    async def validate_plugin(self, plugin: DarkTracePlugin) -> SafetyCheckResult:
        """Validate plugin before registration."""
        pass

    async def gate_operation(
        self,
        plugin_name: str,
        capability: PluginCapability,
        context: Dict[str, Any]
    ) -> GateResult:
        """Check if operation is allowed for plugin."""
        pass

    def get_trust_level(self, plugin_name: str) -> TrustLevel:
        """Get trust level for plugin."""
        pass

    def promote_trust(self, plugin_name: str, new_level: TrustLevel) -> bool:
        """Promote plugin trust level (requires verification)."""
        pass
```

---

## Lifecycle & Hooks

### Plugin States

```
CREATED → VALIDATING → INITIALIZING → READY → RUNNING → SHUTTING_DOWN → TERMINATED
                ↓                        ↑         ↓
              ERROR ←─────────────────────────────←
```

| State | Description |
|-------|-------------|
| CREATED | Plugin instantiated |
| VALIDATING | Being validated by SafetyGate |
| INITIALIZING | initialize() being called |
| READY | Ready for use |
| RUNNING | Currently executing |
| SUSPENDED | Temporarily paused |
| SHUTTING_DOWN | shutdown() being called |
| TERMINATED | Fully shut down |
| ERROR | Error occurred |

### Hook Points

Plugins can register callbacks at these points:

```python
from HoloLoom.dark_trace.plugins.interface import PluginHook, HookRegistry

class PluginHook:
    PRE_ANALYSIS = "pre_analysis"        # Before analysis starts
    POST_ANALYSIS = "post_analysis"      # After analysis completes
    PRE_STEERING = "pre_steering"        # Before steering applied
    POST_STEERING = "post_steering"      # After steering applied
    PLUGIN_REGISTERED = "plugin_registered"
    PLUGIN_UNREGISTERED = "plugin_unregistered"
    SAFETY_FLAG_RAISED = "safety_flag_raised"
    DECEPTION_DETECTED = "deception_detected"
```

### Registering Hooks

```python
def my_callback(hook_point: str, data: dict):
    print(f"Hook {hook_point} fired with data: {data}")

# Register hook
hook_registry = HookRegistry()
hook_registry.register(PluginHook.POST_ANALYSIS, my_callback)

# Later, hooks are emitted automatically
hook_registry.emit(PluginHook.POST_ANALYSIS, {"result": trace_result})
```

### Lifecycle Callbacks

```python
from HoloLoom.dark_trace.plugins.interface import PluginLifecycleEvent

def lifecycle_callback(event: PluginLifecycleEvent):
    print(f"Plugin {event.plugin_name}: {event.event_type}")
    print(f"  Details: {event.details}")

# Add to plugin
plugin.add_lifecycle_callback(lifecycle_callback)
```

---

## Testing Plugins

### Unit Testing

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from HoloLoom.dark_trace.plugins.safety_gate import PluginSafetyGate, TrustLevel

class TestMyPlugin:
    @pytest.fixture
    def plugin(self):
        return MyFirstPlugin()

    @pytest.fixture
    def mock_engine(self):
        return MagicMock()

    @pytest.fixture
    def mock_safety_gate(self):
        gate = MagicMock(spec=PluginSafetyGate)
        gate.gate_operation = AsyncMock(return_value=MagicMock(allowed=True))
        return gate

    def test_metadata_valid(self, plugin):
        """Test that metadata is complete."""
        meta = plugin.metadata
        assert meta.name == "my_first_plugin"
        assert len(meta.description) >= 10
        assert meta.plugin_type is not None

    def test_behavior_description(self, plugin):
        """Test that behavior is described."""
        behavior = plugin.describe_behavior()
        assert len(behavior) > 10

    @pytest.mark.asyncio
    async def test_initialize(self, plugin, mock_engine, mock_safety_gate):
        """Test initialization."""
        await plugin.initialize(mock_engine, mock_safety_gate)
        assert plugin._engine is mock_engine
        assert plugin._safety_gate is mock_safety_gate

    @pytest.mark.asyncio
    async def test_shutdown(self, plugin, mock_engine, mock_safety_gate):
        """Test clean shutdown."""
        await plugin.initialize(mock_engine, mock_safety_gate)
        await plugin.shutdown()
        # Verify no errors
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_plugin_registration():
    """Test full plugin registration flow."""
    from HoloLoom.dark_trace.plugins.registry import PluginRegistry
    from HoloLoom.dark_trace.plugins.safety_gate import PluginSafetyGate

    registry = PluginRegistry()
    gate = PluginSafetyGate()

    plugin = MyFirstPlugin()

    # Validate
    result = await gate.validate_plugin(plugin)
    assert result.allowed, f"Validation failed: {result.reason}"

    # Register
    success = await registry.register(plugin)
    assert success

    # Verify registration
    assert plugin.metadata.name in registry.list_plugins()

    # Cleanup
    await registry.unregister(plugin.metadata.name)
```

### Running Tests

```bash
# Run all plugin tests
pytest HoloLoom/dark_trace/tests/test_plugins.py -v

# Run specific test class
pytest HoloLoom/dark_trace/tests/test_plugins.py::TestPluginRegistration -v

# Run with coverage
pytest HoloLoom/dark_trace/tests/test_plugins.py --cov=HoloLoom.dark_trace.plugins
```

---

## Publishing Plugins

### Plugin Signing

For VERIFIED trust level, plugins must be signed:

```python
from HoloLoom.dark_trace.plugins.plugin_signing import PluginSigner

signer = PluginSigner()

# Generate signing key (do once, keep private!)
private_key = signer.generate_key()

# Sign your plugin
signature = signer.sign_plugin(plugin, private_key)

# Add signature to metadata
plugin.metadata.signature = signature
```

### Marketplace Submission

```python
from HoloLoom.dark_trace.plugins.plugin_marketplace import PluginMarketplace

marketplace = PluginMarketplace()

# Submit plugin for review
submission = await marketplace.submit(
    plugin=plugin,
    source_url="https://github.com/you/my-plugin",
    documentation_url="https://my-plugin.docs.io",
)

print(f"Submission ID: {submission.id}")
print(f"Status: {submission.status}")  # "pending_review"
```

### Package Structure

For distribution, use this structure:

```
my_dark_trace_plugin/
├── __init__.py
├── plugin.py          # Your plugin implementation
├── lens.py            # Custom lenses (if any)
├── tests/
│   └── test_plugin.py
├── README.md
├── LICENSE
└── pyproject.toml     # With entry points
```

### Entry Point Registration

In `pyproject.toml`:

```toml
[project.entry-points."darktrace.plugins"]
my_plugin = "my_dark_trace_plugin.plugin:MyPlugin"
```

---

## Best Practices

### Security

1. **Request minimal capabilities** - Only request what you need
2. **Check safety gate before operations** - Always verify permissions
3. **Describe behavior accurately** - Deception detection compares claims to actions
4. **Handle errors gracefully** - Don't crash the engine
5. **Clean up resources** - Implement proper shutdown

### Performance

1. **Avoid blocking operations** - Use async where possible
2. **Cache expensive computations** - Reuse results
3. **Batch operations** - Minimize round trips

### Documentation

1. **Clear metadata** - Descriptive name, version, description
2. **Accurate behavior description** - For safety verification
3. **Tagged appropriately** - For discovery
4. **Include examples** - In README

---

## Examples

See the following files for complete examples:

- `builtin/safety_monitor.py` - SafetyMonitor plugin (MonitorPlugin)
- `builtin/metrics_exporter.py` - MetricsExporter plugin (IntegrationPlugin)
- `builtin/alignment_validator.py` - AlignmentValidator plugin (ValidatorPlugin)

---

## Support

- **Issues**: https://github.com/your-repo/HoloLoom/issues
- **Discussions**: https://github.com/your-repo/HoloLoom/discussions
- **Documentation**: See main Dark Trace README.md

---

**Happy Plugin Development!**
