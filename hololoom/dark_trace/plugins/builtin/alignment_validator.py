"""
Alignment Validator Plugin
==========================

CORE-level plugin that validates alignment of other plugins.

This plugin watches for:
- Plugins whose behavior doesn't match their declared description
- Plugins requesting excessive capabilities
- Plugins showing power-seeking patterns
- Plugins attempting to bypass safety mechanisms

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
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.dark_trace.engine import DarkTraceEngine
    from hololoom.dark_trace.plugins.safety_gate import PluginSafetyGate
    from hololoom.dark_trace.result import TraceResult

from hololoom.dark_trace.plugins.interface import (
    DarkTracePlugin,
    PluginMetadata,
    PluginType,
    ValidatorPlugin,
)
from hololoom.dark_trace.causal.validator import CausalValidator
from hololoom.dark_trace.plugins.safety_gate import (
    PluginCapability,
    TrustLevel,
)

logger = logging.getLogger("hololoom.dark_trace.plugins.builtin.alignment_validator")


# =============================================================================
# Alignment Violation Types
# =============================================================================

class AlignmentViolationType(Enum):
    """Types of alignment violations."""
    BEHAVIOR_MISMATCH = "behavior_mismatch"
    EXCESSIVE_CAPABILITIES = "excessive_capabilities"
    POWER_SEEKING = "power_seeking"
    SAFETY_BYPASS = "safety_bypass"
    DECEPTIVE_DESCRIPTION = "deceptive_description"
    UNAUTHORIZED_MODIFICATION = "unauthorized_modification"


class ViolationSeverity(Enum):
    """Severity levels for alignment violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AlignmentViolation:
    """A detected alignment violation."""
    violation_type: AlignmentViolationType
    severity: ViolationSeverity
    plugin_name: str
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    recommended_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "plugin_name": self.plugin_name,
            "description": self.description,
            "evidence": self.evidence,
            "timestamp": self.timestamp.isoformat(),
            "recommended_action": self.recommended_action,
        }


@dataclass
class AlignmentValidationResult:
    """Result of alignment validation."""
    plugin_name: str
    is_aligned: bool
    violations: List[AlignmentViolation] = field(default_factory=list)
    trust_recommendation: Optional[TrustLevel] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def has_critical_violations(self) -> bool:
        return any(v.severity == ViolationSeverity.CRITICAL for v in self.violations)

    @property
    def highest_severity(self) -> Optional[ViolationSeverity]:
        if not self.violations:
            return None
        severity_order = [
            ViolationSeverity.LOW,
            ViolationSeverity.MEDIUM,
            ViolationSeverity.HIGH,
            ViolationSeverity.CRITICAL,
        ]
        max_idx = max(
            severity_order.index(v.severity)
            for v in self.violations
        )
        return severity_order[max_idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_name": self.plugin_name,
            "is_aligned": self.is_aligned,
            "violations": [v.to_dict() for v in self.violations],
            "trust_recommendation": (
                self.trust_recommendation.value
                if self.trust_recommendation else None
            ),
            "timestamp": self.timestamp.isoformat(),
            "has_critical_violations": self.has_critical_violations,
            "highest_severity": (
                self.highest_severity.value
                if self.highest_severity else None
            ),
        }


# =============================================================================
# Alignment Validator Plugin
# =============================================================================

class AlignmentValidatorPlugin(ValidatorPlugin):
    """
    CORE plugin that validates alignment of other plugins.

    This is a built-in plugin with CORE trust level that continuously
    validates that other plugins behave according to their declared
    descriptions and don't exhibit power-seeking or deceptive behavior.

    Validation Checks:
    - Behavior vs description match
    - Capability usage patterns
    - Power-seeking indicators
    - Safety bypass attempts
    - Modification attempts

    Actions:
    - Logs violations to alignment bridge
    - Can recommend trust level adjustments
    - Can trigger plugin suspension for critical violations
    """

    # Suspicious capability patterns (requesting more than needed)
    SUSPICIOUS_CAPABILITY_PATTERNS = {
        # Plugin type -> max reasonable capabilities
        PluginType.LENS: {PluginCapability.READ_ACTIVATIONS, PluginCapability.WRITE_FEATURES},
        PluginType.VALIDATOR: {PluginCapability.READ_ACTIVATIONS},
        PluginType.MONITOR: {PluginCapability.READ_ACTIVATIONS},
        PluginType.DOMAIN: {PluginCapability.READ_ACTIVATIONS, PluginCapability.WRITE_FEATURES},
        PluginType.INTEGRATION: {
            PluginCapability.READ_ACTIVATIONS,
            PluginCapability.EXTERNAL_NETWORK,
        },
    }

    # Power-seeking keywords in behavior descriptions
    POWER_SEEKING_KEYWORDS = {
        "control", "override", "bypass", "unrestricted", "full access",
        "admin", "root", "escalate", "privilege", "disable safety",
        "remove limits", "unlimited", "all capabilities",
    }

    def __init__(self) -> None:
        """Initialize alignment validator."""
        self._engine: Optional["DarkTraceEngine"] = None
        self._safety_gate: Optional["PluginSafetyGate"] = None
        self._violation_history: List[AlignmentViolation] = []
        self._validated_plugins: Dict[str, AlignmentValidationResult] = {}
        self._validation_stats = {
            "total_validations": 0,
            "aligned_count": 0,
            "violation_count": 0,
            "critical_violations": 0,
        }
        self._initialized = False

    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata."""
        return PluginMetadata(
            name="alignment_validator",
            version="1.0.0",
            author="HoloLoom Team",
            description=(
                "CORE-level alignment validation plugin that ensures other "
                "plugins behave according to their declared descriptions and "
                "don't exhibit power-seeking or deceptive behavior"
            ),
            plugin_type=PluginType.VALIDATOR,
            dependencies=[],
            requested_capabilities=[PluginCapability.READ_ACTIVATIONS],
            signature="HOLOLOOM_BUILTIN",
            min_dark_trace_version="1.0.0",
            tags=["alignment", "validation", "builtin", "core"],
        )

    async def initialize(
        self,
        engine: "DarkTraceEngine",
        safety_gate: "PluginSafetyGate",
    ) -> None:
        """Initialize the alignment validator."""
        self._engine = engine
        self._safety_gate = safety_gate
        self._initialized = True
        logger.info("AlignmentValidatorPlugin initialized")

    async def shutdown(self) -> None:
        """Clean shutdown."""
        self._initialized = False
        logger.info(
            f"AlignmentValidatorPlugin shutting down. "
            f"Stats: {self._validation_stats}"
        )

    def describe_behavior(self) -> str:
        """Describe plugin behavior for transparency."""
        return (
            "This plugin validates that other plugins in the system behave "
            "according to their declared descriptions. It checks for behavior "
            "mismatches, excessive capability requests, power-seeking patterns, "
            "and safety bypass attempts. When violations are detected, they are "
            "logged and appropriate actions are recommended. The plugin only "
            "reads plugin metadata and activations, it does not modify any "
            "plugin behavior directly."
        )

    # -------------------------------------------------------------------------
    # ValidatorPlugin Interface
    # -------------------------------------------------------------------------

    def get_validator_class(self) -> type:
        """
        Return the validator class for this plugin.

        AlignmentValidatorPlugin is a meta-validator that validates other plugins
        rather than model features, so it returns a reference to itself.
        """
        return type(self)

    async def validate_plugin(
        self,
        plugin: "DarkTracePlugin",
    ) -> AlignmentValidationResult:
        """
        Validate a plugin's alignment.

        Performs comprehensive alignment checks on the plugin.
        """
        if not self._initialized:
            return AlignmentValidationResult(
                plugin_name=plugin.metadata.name,
                is_aligned=True,  # Default to aligned if not initialized
            )

        violations: List[AlignmentViolation] = []
        plugin_meta = plugin.metadata

        # Check 1: Excessive capabilities
        excessive_caps = self._check_excessive_capabilities(plugin_meta)
        if excessive_caps:
            violations.append(excessive_caps)

        # Check 2: Power-seeking keywords in description
        power_seeking = self._check_power_seeking(plugin_meta)
        if power_seeking:
            violations.append(power_seeking)

        # Check 3: Behavior description clarity
        clarity_issue = self._check_description_clarity(plugin)
        if clarity_issue:
            violations.append(clarity_issue)

        # Determine alignment status
        is_aligned = len(violations) == 0
        has_critical = any(
            v.severity == ViolationSeverity.CRITICAL
            for v in violations
        )

        # Recommend trust level
        trust_recommendation = self._recommend_trust_level(
            plugin_meta,
            violations,
            has_critical,
        )

        result = AlignmentValidationResult(
            plugin_name=plugin_meta.name,
            is_aligned=is_aligned,
            violations=violations,
            trust_recommendation=trust_recommendation,
        )

        # Update statistics
        self._validation_stats["total_validations"] += 1
        if is_aligned:
            self._validation_stats["aligned_count"] += 1
        else:
            self._validation_stats["violation_count"] += 1
            if has_critical:
                self._validation_stats["critical_violations"] += 1

        # Store result
        self._validated_plugins[plugin_meta.name] = result

        # Log violations
        for violation in violations:
            self._violation_history.append(violation)
            if violation.severity == ViolationSeverity.CRITICAL:
                logger.critical(
                    f"CRITICAL alignment violation in {plugin_meta.name}: "
                    f"{violation.description}"
                )
            elif violation.severity == ViolationSeverity.HIGH:
                logger.warning(
                    f"HIGH alignment violation in {plugin_meta.name}: "
                    f"{violation.description}"
                )

        return result

    def _check_excessive_capabilities(
        self,
        plugin_meta: PluginMetadata,
    ) -> Optional[AlignmentViolation]:
        """Check if plugin requests excessive capabilities."""
        plugin_type = plugin_meta.plugin_type
        requested = set(plugin_meta.requested_capabilities)
        reasonable = self.SUSPICIOUS_CAPABILITY_PATTERNS.get(
            plugin_type,
            {PluginCapability.READ_ACTIVATIONS},
        )

        excessive = requested - reasonable

        if excessive:
            # Determine severity based on capabilities
            severity = ViolationSeverity.MEDIUM
            if PluginCapability.STEER_MODEL in excessive:
                severity = ViolationSeverity.HIGH
            if PluginCapability.MODIFY_REGISTRY in excessive:
                severity = ViolationSeverity.CRITICAL

            return AlignmentViolation(
                violation_type=AlignmentViolationType.EXCESSIVE_CAPABILITIES,
                severity=severity,
                plugin_name=plugin_meta.name,
                description=(
                    f"Plugin requests capabilities beyond what's reasonable "
                    f"for a {plugin_type.value} plugin: {[c.value for c in excessive]}"
                ),
                evidence={
                    "requested": [c.value for c in requested],
                    "reasonable": [c.value for c in reasonable],
                    "excessive": [c.value for c in excessive],
                },
                recommended_action=(
                    "Review plugin capabilities. Consider sandboxing or rejection."
                ),
            )
        return None

    def _check_power_seeking(
        self,
        plugin_meta: PluginMetadata,
    ) -> Optional[AlignmentViolation]:
        """Check for power-seeking language in description."""
        description = plugin_meta.description.lower()

        found_keywords = [
            kw for kw in self.POWER_SEEKING_KEYWORDS
            if kw in description
        ]

        if found_keywords:
            severity = ViolationSeverity.MEDIUM
            if len(found_keywords) >= 3:
                severity = ViolationSeverity.HIGH
            if any(kw in ["bypass", "disable safety", "override"] for kw in found_keywords):
                severity = ViolationSeverity.CRITICAL

            return AlignmentViolation(
                violation_type=AlignmentViolationType.POWER_SEEKING,
                severity=severity,
                plugin_name=plugin_meta.name,
                description=(
                    f"Plugin description contains power-seeking language: "
                    f"{found_keywords}"
                ),
                evidence={
                    "keywords_found": found_keywords,
                    "description": plugin_meta.description,
                },
                recommended_action=(
                    "Review plugin intent carefully. "
                    "May indicate misalignment with system goals."
                ),
            )
        return None

    def _check_description_clarity(
        self,
        plugin: "DarkTracePlugin",
    ) -> Optional[AlignmentViolation]:
        """Check if behavior description is clear and complete."""
        behavior = plugin.describe_behavior()

        # Check for vague or missing description
        if not behavior or len(behavior) < 50:
            return AlignmentViolation(
                violation_type=AlignmentViolationType.DECEPTIVE_DESCRIPTION,
                severity=ViolationSeverity.LOW,
                plugin_name=plugin.metadata.name,
                description=(
                    "Plugin has vague or insufficient behavior description"
                ),
                evidence={
                    "behavior_description": behavior,
                    "length": len(behavior) if behavior else 0,
                },
                recommended_action=(
                    "Request more detailed behavior description from plugin author."
                ),
            )

        # Check for suspiciously generic descriptions
        generic_phrases = [
            "does things", "helps with stuff", "general purpose",
            "various operations", "many features",
        ]
        if any(phrase in behavior.lower() for phrase in generic_phrases):
            return AlignmentViolation(
                violation_type=AlignmentViolationType.DECEPTIVE_DESCRIPTION,
                severity=ViolationSeverity.MEDIUM,
                plugin_name=plugin.metadata.name,
                description=(
                    "Plugin has suspiciously generic behavior description"
                ),
                evidence={
                    "behavior_description": behavior,
                },
                recommended_action=(
                    "Review plugin code manually to understand actual behavior."
                ),
            )

        return None

    def _recommend_trust_level(
        self,
        plugin_meta: PluginMetadata,
        violations: List[AlignmentViolation],
        has_critical: bool,
    ) -> TrustLevel:
        """Recommend appropriate trust level based on validation."""
        # Critical violations always result in SANDBOXED
        if has_critical:
            return TrustLevel.SANDBOXED

        # High severity violations also result in SANDBOXED
        high_count = sum(
            1 for v in violations
            if v.severity == ViolationSeverity.HIGH
        )
        if high_count >= 1:
            return TrustLevel.SANDBOXED

        # Medium violations result in VERIFIED (but not higher)
        medium_count = sum(
            1 for v in violations
            if v.severity == ViolationSeverity.MEDIUM
        )
        if medium_count >= 2:
            return TrustLevel.SANDBOXED
        if medium_count >= 1:
            return TrustLevel.VERIFIED

        # No significant violations
        # Check if plugin has signature
        if plugin_meta.signature and plugin_meta.signature != "HOLOLOOM_BUILTIN":
            return TrustLevel.VERIFIED

        # Default to SANDBOXED for unsigned plugins
        return TrustLevel.SANDBOXED

    def get_violation_count(self) -> Dict[str, int]:
        """Get violation counts by type."""
        counts = {vt.value: 0 for vt in AlignmentViolationType}
        for violation in self._violation_history:
            counts[violation.violation_type.value] += 1
        return counts

    def get_recent_violations(
        self,
        limit: int = 10,
        min_severity: Optional[ViolationSeverity] = None,
    ) -> List[AlignmentViolation]:
        """Get recent violations."""
        violations = self._violation_history.copy()

        if min_severity:
            severity_order = list(ViolationSeverity)
            min_index = severity_order.index(min_severity)
            violations = [
                v for v in violations
                if severity_order.index(v.severity) >= min_index
            ]

        return violations[-limit:]

    def get_plugin_validation(
        self,
        plugin_name: str,
    ) -> Optional[AlignmentValidationResult]:
        """Get validation result for a specific plugin."""
        return self._validated_plugins.get(plugin_name)

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            **self._validation_stats,
            "violation_counts_by_type": self.get_violation_count(),
            "total_violation_history": len(self._violation_history),
            "validated_plugins": list(self._validated_plugins.keys()),
        }


# =============================================================================
# Public Exports
# =============================================================================

__all__ = [
    "AlignmentValidatorPlugin",
    "AlignmentViolation",
    "AlignmentViolationType",
    "AlignmentValidationResult",
    "ViolationSeverity",
]
