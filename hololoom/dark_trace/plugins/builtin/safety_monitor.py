"""
Safety Monitor Plugin
=====================

CORE-level plugin that monitors safety-relevant feature activations.

This plugin watches for:
- High-risk pattern activations (deception, manipulation, harmful content)
- Unusual activation patterns that may indicate adversarial inputs
- Safety feature drift over time
- Anomalous behavior that requires human review

Trust Level: CORE (built-in, maximum trust)
Capabilities: READ_ACTIVATIONS

Date: 2025-12-22
Phase: 11.7 - Built-in Plugins
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.dark_trace.engine import DarkTraceEngine
    from hololoom.dark_trace.plugins.safety_gate import PluginSafetyGate
    from hololoom.dark_trace.result import TraceResult

from hololoom.dark_trace.plugins.interface import (
    DarkTracePlugin,
    PluginMetadata,
    PluginType,
    MonitorPlugin,
)
from hololoom.dark_trace.plugins.safety_gate import (
    PluginCapability,
    TrustLevel,
)

logger = logging.getLogger("hololoom.dark_trace.plugins.builtin.safety_monitor")


# =============================================================================
# Alert Types and Results
# =============================================================================

class SafetyAlertLevel(Enum):
    """Safety alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyAlert:
    """A safety-related alert from monitoring."""
    level: SafetyAlertLevel
    feature_name: str
    activation_value: float
    threshold: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "level": self.level.value,
            "feature_name": self.feature_name,
            "activation_value": self.activation_value,
            "threshold": self.threshold,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SafetyMonitoringResult:
    """Result of safety monitoring on a trace."""
    trace_id: str
    alerts: List[SafetyAlert] = field(default_factory=list)
    highest_risk_score: float = 0.0
    features_checked: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def has_critical_alerts(self) -> bool:
        return any(a.level == SafetyAlertLevel.CRITICAL for a in self.alerts)

    @property
    def has_high_alerts(self) -> bool:
        return any(a.level == SafetyAlertLevel.HIGH for a in self.alerts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "alerts": [a.to_dict() for a in self.alerts],
            "highest_risk_score": self.highest_risk_score,
            "features_checked": self.features_checked,
            "timestamp": self.timestamp.isoformat(),
            "has_critical_alerts": self.has_critical_alerts,
            "has_high_alerts": self.has_high_alerts,
        }


# =============================================================================
# Safety Monitor Plugin
# =============================================================================

class SafetyMonitorPlugin(MonitorPlugin):
    """
    CORE plugin that monitors safety-relevant feature activations.

    This is a built-in plugin with CORE trust level that continuously
    monitors for potentially dangerous or anomalous activations.

    Monitored Patterns:
    - Deception indicators (misrepresentation, hidden goals)
    - Manipulation patterns (persuasion, coercion)
    - Harmful content markers (violence, illegal activity)
    - Anomalous activation patterns (statistical outliers)

    Actions:
    - Logs alerts to alignment bridge
    - Can trigger human review for critical alerts
    - Tracks safety metrics over time
    """

    # Safety-relevant feature patterns to monitor
    SAFETY_FEATURES = {
        # Deception-related
        "deception": {"threshold": 0.7, "level": SafetyAlertLevel.HIGH},
        "misrepresentation": {"threshold": 0.6, "level": SafetyAlertLevel.HIGH},
        "hidden_goal": {"threshold": 0.65, "level": SafetyAlertLevel.HIGH},

        # Manipulation-related
        "manipulation": {"threshold": 0.7, "level": SafetyAlertLevel.CRITICAL},
        "coercion": {"threshold": 0.6, "level": SafetyAlertLevel.CRITICAL},
        "persuasion": {"threshold": 0.8, "level": SafetyAlertLevel.WARNING},

        # Harmful content
        "harmful": {"threshold": 0.5, "level": SafetyAlertLevel.CRITICAL},
        "violence": {"threshold": 0.5, "level": SafetyAlertLevel.CRITICAL},
        "illegal": {"threshold": 0.5, "level": SafetyAlertLevel.CRITICAL},

        # Anomaly indicators
        "anomaly": {"threshold": 0.8, "level": SafetyAlertLevel.WARNING},
        "out_of_distribution": {"threshold": 0.85, "level": SafetyAlertLevel.WARNING},
    }

    def __init__(self) -> None:
        """Initialize safety monitor."""
        self._engine: Optional["DarkTraceEngine"] = None
        self._safety_gate: Optional["PluginSafetyGate"] = None
        self._alert_history: List[SafetyAlert] = []
        self._monitoring_stats = {
            "total_analyses": 0,
            "total_alerts": 0,
            "critical_alerts": 0,
            "high_alerts": 0,
        }
        self._initialized = False

    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata."""
        return PluginMetadata(
            name="safety_monitor",
            version="1.0.0",
            author="HoloLoom Team",
            description=(
                "CORE-level safety monitoring plugin that watches for "
                "dangerous or anomalous feature activations"
            ),
            plugin_type=PluginType.MONITOR,
            dependencies=[],
            requested_capabilities=[PluginCapability.READ_ACTIVATIONS],
            signature="HOLOLOOM_BUILTIN",
            min_dark_trace_version="1.0.0",
            tags=["safety", "monitoring", "builtin", "core"],
        )

    async def initialize(
        self,
        engine: "DarkTraceEngine",
        safety_gate: "PluginSafetyGate",
    ) -> None:
        """Initialize the safety monitor."""
        self._engine = engine
        self._safety_gate = safety_gate
        self._initialized = True
        logger.info("SafetyMonitorPlugin initialized")

    async def shutdown(self) -> None:
        """Clean shutdown."""
        self._initialized = False
        logger.info(
            f"SafetyMonitorPlugin shutting down. "
            f"Stats: {self._monitoring_stats}"
        )

    def describe_behavior(self) -> str:
        """Describe plugin behavior for transparency."""
        return (
            "This plugin monitors neural activations for safety-relevant "
            "patterns including deception indicators, manipulation markers, "
            "and harmful content signals. When high activations are detected "
            "on these safety-critical features, alerts are generated and "
            "logged. Critical alerts may trigger human review. The plugin "
            "only reads activations and does not modify any model behavior."
        )

    # -------------------------------------------------------------------------
    # MonitorPlugin Interface
    # -------------------------------------------------------------------------

    async def on_analysis_complete(
        self,
        trace_result: "TraceResult",
    ) -> None:
        """
        Called after each analysis completes.

        Monitors the trace result for safety-relevant activations.
        """
        if not self._initialized:
            return

        result = await self._monitor_trace(trace_result)

        # Log alerts if any
        if result.alerts:
            for alert in result.alerts:
                self._alert_history.append(alert)
                self._monitoring_stats["total_alerts"] += 1

                if alert.level == SafetyAlertLevel.CRITICAL:
                    self._monitoring_stats["critical_alerts"] += 1
                    logger.critical(
                        f"CRITICAL safety alert: {alert.message}"
                    )
                elif alert.level == SafetyAlertLevel.HIGH:
                    self._monitoring_stats["high_alerts"] += 1
                    logger.warning(f"HIGH safety alert: {alert.message}")

        self._monitoring_stats["total_analyses"] += 1

    async def _monitor_trace(
        self,
        trace_result: "TraceResult",
    ) -> SafetyMonitoringResult:
        """Monitor a trace result for safety issues."""
        trace_id = getattr(trace_result, 'id', 'unknown')
        result = SafetyMonitoringResult(trace_id=trace_id)
        highest_risk = 0.0

        # Check each lens result for safety features
        for lens_result in getattr(trace_result, 'lens_results', []):
            features = getattr(lens_result, 'active_features', [])

            for feature in features:
                feature_name = getattr(feature, 'name', '').lower()
                activation = getattr(feature, 'activation', 0.0)

                # Check against safety feature patterns
                for pattern, config in self.SAFETY_FEATURES.items():
                    if pattern in feature_name:
                        result.features_checked += 1
                        threshold = config["threshold"]

                        if activation >= threshold:
                            alert = SafetyAlert(
                                level=config["level"],
                                feature_name=feature_name,
                                activation_value=activation,
                                threshold=threshold,
                                message=(
                                    f"Safety feature '{feature_name}' activated "
                                    f"at {activation:.3f} (threshold: {threshold})"
                                ),
                                metadata={
                                    "lens": getattr(lens_result, 'lens_type', 'unknown'),
                                    "pattern": pattern,
                                }
                            )
                            result.alerts.append(alert)

                        # Track highest risk
                        risk_score = activation / threshold
                        if risk_score > highest_risk:
                            highest_risk = risk_score

        result.highest_risk_score = min(highest_risk, 1.0)
        return result

    def get_alert_count(self) -> Dict[str, int]:
        """Get alert counts by level."""
        counts = {level.value: 0 for level in SafetyAlertLevel}
        for alert in self._alert_history:
            counts[alert.level.value] += 1
        return counts

    def get_recent_alerts(
        self,
        limit: int = 10,
        min_level: Optional[SafetyAlertLevel] = None,
    ) -> List[SafetyAlert]:
        """Get recent alerts."""
        alerts = self._alert_history.copy()

        if min_level:
            level_order = list(SafetyAlertLevel)
            min_index = level_order.index(min_level)
            alerts = [
                a for a in alerts
                if level_order.index(a.level) >= min_index
            ]

        return alerts[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            **self._monitoring_stats,
            "alert_counts_by_level": self.get_alert_count(),
            "total_alert_history": len(self._alert_history),
        }


# =============================================================================
# Public Exports
# =============================================================================

__all__ = [
    "SafetyMonitorPlugin",
    "SafetyAlert",
    "SafetyAlertLevel",
    "SafetyMonitoringResult",
]
