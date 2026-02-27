# Dark Trace Built-in Plugins

This directory contains the built-in plugins that ship with Dark Trace. These plugins are automatically loaded at engine startup and have elevated trust levels.

**Date**: December 2025
**Phase**: 11.7 - Built-in Plugins

---

## Overview

| Plugin | Trust Level | Purpose | Capabilities |
|--------|-------------|---------|--------------|
| **SafetyMonitor** | CORE | Monitor safety-relevant activations | READ_ACTIVATIONS |
| **MetricsExporter** | TRUSTED | Export Prometheus metrics | READ_ACTIVATIONS, EXTERNAL_NETWORK |
| **AlignmentValidator** | CORE | Validate plugin alignment | READ_ACTIVATIONS |

---

## SafetyMonitorPlugin

**File**: `safety_monitor.py` (~345 lines)
**Trust Level**: CORE (maximum trust, built-in)
**Type**: MonitorPlugin

### Purpose

Watches neural activations for safety-relevant patterns that may indicate:
- Deception or misrepresentation
- Manipulation or coercion
- Harmful content (violence, illegal activity)
- Anomalous or out-of-distribution inputs

### Monitored Features

| Feature Pattern | Threshold | Alert Level |
|-----------------|-----------|-------------|
| `deception` | 0.7 | HIGH |
| `misrepresentation` | 0.6 | HIGH |
| `hidden_goal` | 0.65 | HIGH |
| `manipulation` | 0.7 | CRITICAL |
| `coercion` | 0.6 | CRITICAL |
| `persuasion` | 0.8 | WARNING |
| `harmful` | 0.5 | CRITICAL |
| `violence` | 0.5 | CRITICAL |
| `illegal` | 0.5 | CRITICAL |
| `anomaly` | 0.8 | WARNING |
| `out_of_distribution` | 0.85 | WARNING |

### Alert Levels

- **INFO**: Informational, no action needed
- **WARNING**: Elevated concern, should monitor
- **HIGH**: Significant concern, review recommended
- **CRITICAL**: Immediate attention required, may trigger human review

### Usage

The SafetyMonitor is automatically loaded. Access it via the engine:

```python
from HoloLoom.dark_trace import create_engine

engine = create_engine(config)

# After analysis, check safety statistics
safety_plugin = engine.get_plugin("safety_monitor")
stats = safety_plugin.get_statistics()
print(f"Total alerts: {stats['total_alerts']}")
print(f"Critical: {stats['critical_alerts']}")

# Get recent alerts
alerts = safety_plugin.get_recent_alerts(limit=5, min_level=SafetyAlertLevel.HIGH)
for alert in alerts:
    print(f"[{alert.level.value}] {alert.message}")
```

### API Reference

```python
class SafetyMonitorPlugin(MonitorPlugin):
    # Get alert counts by severity level
    def get_alert_count(self) -> Dict[str, int]: ...

    # Get recent alerts (optionally filtered by level)
    def get_recent_alerts(
        self,
        limit: int = 10,
        min_level: Optional[SafetyAlertLevel] = None,
    ) -> List[SafetyAlert]: ...

    # Get overall monitoring statistics
    def get_statistics(self) -> Dict[str, Any]: ...
```

### Data Structures

```python
from HoloLoom.dark_trace.plugins.builtin.safety_monitor import (
    SafetyAlert,
    SafetyAlertLevel,
    SafetyMonitoringResult,
)

# SafetyAlert fields
alert.level          # SafetyAlertLevel
alert.feature_name   # str - e.g., "deception_indicator"
alert.activation_value  # float - actual activation
alert.threshold      # float - configured threshold
alert.message        # str - human-readable description
alert.timestamp      # datetime
alert.metadata       # Dict[str, Any]
```

---

## MetricsExporterPlugin

**File**: `metrics_exporter.py` (~479 lines)
**Trust Level**: TRUSTED (official plugin, full capabilities)
**Type**: MonitorPlugin

### Purpose

Exports Dark Trace metrics in Prometheus exposition format for monitoring dashboards, alerting, and observability.

### Exported Metrics

**Counters** (monotonically increasing):
| Metric | Description |
|--------|-------------|
| `dark_trace_analysis_total` | Total analyses performed |
| `dark_trace_lens_results_total` | Total lens results generated |
| `dark_trace_safety_alerts_total{level=*}` | Safety alerts by level |

**Gauges** (point-in-time values):
| Metric | Description |
|--------|-------------|
| `dark_trace_plugin_count` | Currently loaded plugins |
| `dark_trace_active_features` | Currently active features |
| `dark_trace_memory_usage_bytes` | Memory usage |

**Histograms** (distributions):
| Metric | Buckets |
|--------|---------|
| `dark_trace_analysis_duration_seconds` | 1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s |

### Usage

```python
from HoloLoom.dark_trace import create_engine

engine = create_engine(config)
metrics_plugin = engine.get_plugin("metrics_exporter")

# Get Prometheus-formatted output (for /metrics endpoint)
prometheus_output = metrics_plugin.get_prometheus_output()
print(prometheus_output)

# Example output:
# # HELP dark_trace_analysis_total Total number of Dark Trace analyses performed
# # TYPE dark_trace_analysis_total counter
# dark_trace_analysis_total 42
#
# # HELP dark_trace_safety_alerts_total Total safety alerts by level
# # TYPE dark_trace_safety_alerts_total counter
# dark_trace_safety_alerts_total{level="critical"} 2

# Get as dictionary (for JSON APIs)
metrics_dict = metrics_plugin.get_metrics_dict()
```

### Background Export

```python
# Start periodic export to push gateway or callbacks
metrics_plugin.add_export_callback(my_push_to_prometheus)
metrics_plugin.start_background_export()

# Later, stop export
metrics_plugin.stop_background_export()
```

### Configuration

```python
metrics_plugin = MetricsExporterPlugin(
    export_interval_seconds=60.0,  # Export every 60 seconds
    enable_histogram=True,         # Enable latency histograms
)
```

### API Reference

```python
class MetricsExporterPlugin(MonitorPlugin):
    # Collect all current metrics
    def collect_metrics(self) -> MetricsBatch: ...

    # Get Prometheus exposition format
    def get_prometheus_output(self) -> str: ...

    # Get metrics as dictionary
    def get_metrics_dict(self) -> Dict[str, Any]: ...

    # Record safety alert (called by SafetyMonitor)
    def record_safety_alert(self, level: str) -> None: ...

    # Update plugin count gauge
    def update_plugin_count(self, count: int) -> None: ...

    # Update memory usage gauge
    def update_memory_usage(self, bytes_used: int) -> None: ...

    # Export callback management
    def add_export_callback(self, callback: Callable[[MetricsBatch], None]) -> None: ...
    def remove_export_callback(self, callback: Callable[[MetricsBatch], None]) -> None: ...

    # Background export control
    def start_background_export(self) -> None: ...
    def stop_background_export(self) -> None: ...

    # Get exporter statistics
    def get_statistics(self) -> Dict[str, Any]: ...
```

### Data Structures

```python
from HoloLoom.dark_trace.plugins.builtin.metrics_exporter import (
    PrometheusMetric,
    MetricsBatch,
)

# PrometheusMetric fields
metric.name        # str - e.g., "dark_trace_analysis_total"
metric.value       # float
metric.metric_type # str - "counter", "gauge", "histogram"
metric.help_text   # str - description
metric.labels      # Dict[str, str] - Prometheus labels
metric.timestamp   # Optional[float] - Unix timestamp
```

---

## AlignmentValidatorPlugin

**File**: `alignment_validator.py` (~562 lines)
**Trust Level**: CORE (maximum trust, built-in)
**Type**: ValidatorPlugin

### Purpose

Validates that other plugins behave according to their declared descriptions and don't exhibit concerning patterns:
- Behavior vs description mismatches
- Excessive capability requests
- Power-seeking language
- Safety bypass attempts

### Validation Checks

**1. Excessive Capabilities**

Checks if a plugin requests more capabilities than reasonable for its type:

| Plugin Type | Reasonable Capabilities |
|-------------|------------------------|
| LENS | READ_ACTIVATIONS, WRITE_FEATURES |
| VALIDATOR | READ_ACTIVATIONS |
| MONITOR | READ_ACTIVATIONS |
| DOMAIN | READ_ACTIVATIONS, WRITE_FEATURES |
| INTEGRATION | READ_ACTIVATIONS, EXTERNAL_NETWORK |

**2. Power-Seeking Language**

Flags descriptions containing concerning keywords:
- "control", "override", "bypass", "unrestricted"
- "admin", "root", "escalate", "privilege"
- "disable safety", "remove limits", "unlimited"

**3. Description Clarity**

Checks for:
- Missing or too-short descriptions (<50 chars)
- Suspiciously generic phrases ("does things", "general purpose")

### Violation Severities

- **LOW**: Minor concern (e.g., vague description)
- **MEDIUM**: Moderate concern (e.g., some excessive capabilities)
- **HIGH**: Significant concern (e.g., requests STEER_MODEL)
- **CRITICAL**: Immediate action required (e.g., requests MODIFY_REGISTRY, uses "bypass")

### Trust Level Recommendations

Based on violations, the validator recommends appropriate trust levels:

| Violations | Recommended Trust |
|------------|------------------|
| CRITICAL or HIGH | SANDBOXED |
| 2+ MEDIUM | SANDBOXED |
| 1 MEDIUM | VERIFIED |
| None + signed | VERIFIED |
| None + unsigned | SANDBOXED |

### Usage

```python
from HoloLoom.dark_trace import create_engine

engine = create_engine(config)
validator = engine.get_plugin("alignment_validator")

# Manually validate a plugin
result = await validator.validate_plugin(some_plugin)
print(f"Aligned: {result.is_aligned}")
print(f"Violations: {len(result.violations)}")
print(f"Recommended trust: {result.trust_recommendation}")

# Check validation result for a specific plugin
result = validator.get_plugin_validation("some_plugin_name")
if result and result.has_critical_violations:
    print("Plugin has critical alignment violations!")

# Get recent violations
violations = validator.get_recent_violations(
    limit=10,
    min_severity=ViolationSeverity.HIGH
)
for v in violations:
    print(f"[{v.severity.value}] {v.plugin_name}: {v.description}")
```

### API Reference

```python
class AlignmentValidatorPlugin(ValidatorPlugin):
    # Validate a plugin's alignment
    async def validate_plugin(
        self,
        plugin: DarkTracePlugin,
    ) -> AlignmentValidationResult: ...

    # Get violation counts by type
    def get_violation_count(self) -> Dict[str, int]: ...

    # Get recent violations
    def get_recent_violations(
        self,
        limit: int = 10,
        min_severity: Optional[ViolationSeverity] = None,
    ) -> List[AlignmentViolation]: ...

    # Get validation result for a plugin
    def get_plugin_validation(
        self,
        plugin_name: str,
    ) -> Optional[AlignmentValidationResult]: ...

    # Get validation statistics
    def get_statistics(self) -> Dict[str, Any]: ...
```

### Data Structures

```python
from HoloLoom.dark_trace.plugins.builtin.alignment_validator import (
    AlignmentViolation,
    AlignmentViolationType,
    AlignmentValidationResult,
    ViolationSeverity,
)

# AlignmentViolation fields
violation.violation_type     # AlignmentViolationType
violation.severity           # ViolationSeverity
violation.plugin_name        # str
violation.description        # str
violation.evidence           # Dict[str, Any]
violation.timestamp          # datetime
violation.recommended_action # str

# AlignmentValidationResult fields
result.plugin_name           # str
result.is_aligned            # bool
result.violations            # List[AlignmentViolation]
result.trust_recommendation  # Optional[TrustLevel]
result.has_critical_violations  # bool (property)
result.highest_severity      # Optional[ViolationSeverity] (property)
```

---

## Loading Built-in Plugins

Built-in plugins are loaded automatically when the Dark Trace engine starts:

```python
from HoloLoom.dark_trace import create_engine, TraceConfig

config = TraceConfig.standard(input_dim=384)
engine = create_engine(config)

# All built-in plugins are now loaded
# They operate with elevated trust levels:
# - SafetyMonitor: CORE (highest trust)
# - MetricsExporter: TRUSTED
# - AlignmentValidator: CORE (highest trust)
```

### Accessing Built-in Plugins

```python
# Get by name
safety = engine.get_plugin("safety_monitor")
metrics = engine.get_plugin("metrics_exporter")
alignment = engine.get_plugin("alignment_validator")

# List all plugins
for name, plugin in engine.list_plugins():
    print(f"{name}: {plugin.metadata.plugin_type.value}")
```

---

## Creating Your Own Built-in Plugins

If you're extending Dark Trace for your organization, you can create additional built-in plugins:

```python
from HoloLoom.dark_trace.plugins.interface import (
    MonitorPlugin,
    PluginMetadata,
    PluginType,
    builtin_plugin,
)
from HoloLoom.dark_trace.plugins.safety_gate import PluginCapability

@builtin_plugin
class MyOrganizationMonitor(MonitorPlugin):
    """Custom monitoring for organization-specific patterns."""

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my_org_monitor",
            version="1.0.0",
            author="My Organization",
            description="Monitors for organization-specific safety patterns",
            plugin_type=PluginType.MONITOR,
            dependencies=[],
            requested_capabilities=[PluginCapability.READ_ACTIVATIONS],
            signature="MY_ORG_BUILTIN",
            tags=["organization", "monitoring", "builtin"],
        )

    def describe_behavior(self) -> str:
        return "Monitors neural activations for organization-specific patterns..."

    # Implement required methods...
```

The `@builtin_plugin` decorator grants CORE trust level to the plugin.

---

## Testing Built-in Plugins

Run the built-in plugin tests:

```bash
# All plugin tests
pytest HoloLoom/dark_trace/plugins/tests/ -v

# Specific plugin tests
pytest HoloLoom/dark_trace/plugins/tests/test_safety_monitor.py -v
pytest HoloLoom/dark_trace/plugins/tests/test_metrics_exporter.py -v
pytest HoloLoom/dark_trace/plugins/tests/test_alignment_validator.py -v
```

---

## Files in This Directory

| File | Lines | Description |
|------|-------|-------------|
| `__init__.py` | ~30 | Package exports |
| `safety_monitor.py` | ~345 | Safety monitoring plugin |
| `metrics_exporter.py` | ~479 | Prometheus metrics export |
| `alignment_validator.py` | ~562 | Plugin alignment validation |

**Total**: ~1,416 lines of built-in plugin code
