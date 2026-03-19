"""
Built-in Plugins for Dark Trace
===============================

This module contains the official built-in plugins for Dark Trace.
These plugins have special trust levels and provide core functionality.

Built-in Plugin Types:
- CORE level: SafetyMonitorPlugin, AlignmentValidatorPlugin
- TRUSTED level: MetricsExporterPlugin

All built-in plugins:
- Have signature "HOLOLOOM_BUILTIN"
- Are pre-registered with appropriate trust levels
- Provide essential safety, validation, and monitoring functionality

Date: 2025-12-22
Phase: 11.7 - Built-in Plugins
"""

from hololoom.dark_trace.plugins.builtin.alignment_validator import (
    AlignmentValidationResult,
    AlignmentValidatorPlugin,
    AlignmentViolation,
    AlignmentViolationType,
    ViolationSeverity,
)
from hololoom.dark_trace.plugins.builtin.metrics_exporter import (
    MetricsBatch,
    MetricsExporterPlugin,
    PrometheusMetric,
)
from hololoom.dark_trace.plugins.builtin.safety_monitor import (
    SafetyAlert,
    SafetyAlertLevel,
    SafetyMonitoringResult,
    SafetyMonitorPlugin,
)

# =============================================================================
# Built-in Plugin Registry
# =============================================================================

# All built-in plugins with their trust levels
BUILTIN_PLUGINS = {
    "safety_monitor": {
        "class": SafetyMonitorPlugin,
        "trust_level": "core",
        "description": "CORE-level safety monitoring plugin",
        "auto_load": True,
    },
    "alignment_validator": {
        "class": AlignmentValidatorPlugin,
        "trust_level": "core",
        "description": "CORE-level alignment validation plugin",
        "auto_load": True,
    },
    "metrics_exporter": {
        "class": MetricsExporterPlugin,
        "trust_level": "trusted",
        "description": "TRUSTED-level Prometheus metrics exporter",
        "auto_load": False,  # Requires explicit configuration
    },
}


def get_builtin_plugin(name: str):
    """
    Get a built-in plugin class by name.

    Args:
        name: Plugin name (e.g., "safety_monitor")

    Returns:
        Plugin class or None if not found
    """
    plugin_info = BUILTIN_PLUGINS.get(name)
    if plugin_info:
        return plugin_info["class"]
    return None


def get_auto_load_plugins():
    """
    Get list of plugins that should be auto-loaded.

    Returns:
        List of (name, class) tuples for auto-load plugins
    """
    return [
        (name, info["class"])
        for name, info in BUILTIN_PLUGINS.items()
        if info.get("auto_load", False)
    ]


def list_builtin_plugins():
    """
    List all available built-in plugins.

    Returns:
        Dict of plugin name -> plugin info
    """
    return {
        name: {
            "trust_level": info["trust_level"],
            "description": info["description"],
            "auto_load": info.get("auto_load", False),
        }
        for name, info in BUILTIN_PLUGINS.items()
    }


# =============================================================================
# Public Exports
# =============================================================================

__all__ = [
    # Safety Monitor
    "SafetyMonitorPlugin",
    "SafetyAlert",
    "SafetyAlertLevel",
    "SafetyMonitoringResult",
    # Alignment Validator
    "AlignmentValidatorPlugin",
    "AlignmentViolation",
    "AlignmentViolationType",
    "AlignmentValidationResult",
    "ViolationSeverity",
    # Metrics Exporter
    "MetricsExporterPlugin",
    "PrometheusMetric",
    "MetricsBatch",
    # Registry functions
    "BUILTIN_PLUGINS",
    "get_builtin_plugin",
    "get_auto_load_plugins",
    "list_builtin_plugins",
]
