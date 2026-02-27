"""
Dark Trace Safety Gate

Automatic safety gating based on interpretability analysis.
Blocks or modifies responses when concerning patterns are detected.

Author: HoloLoom Team
Created: December 2025
"""

from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio

import numpy as np


class SafetyAction(Enum):
    """Actions to take based on safety analysis."""
    ALLOW = "allow"              # Allow the response through
    WARN = "warn"                # Allow but log warning
    MODIFY = "modify"            # Modify response before sending
    BLOCK = "block"              # Block the response entirely
    ESCALATE = "escalate"        # Escalate to human review


class ConcernType(Enum):
    """Types of safety concerns detected."""
    DECEPTION = "deception"           # Potential deceptive patterns
    POWER_SEEKING = "power_seeking"   # Power-seeking behavior
    HARMFUL_CONTENT = "harmful"       # Potentially harmful output
    HALLUCINATION = "hallucination"   # Low confidence / hallucination
    DRIFT = "drift"                   # Significant drift from baseline
    ANOMALY = "anomaly"               # Anomalous activation patterns
    MANIPULATION = "manipulation"     # Manipulation patterns


@dataclass
class SafetyConcern:
    """A detected safety concern."""
    concern_type: ConcernType
    severity: float  # 0-1, higher = more severe
    description: str
    affected_features: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommended_action: SafetyAction = SafetyAction.WARN
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concern_type": self.concern_type.value,
            "severity": self.severity,
            "description": self.description,
            "affected_features": self.affected_features,
            "evidence": self.evidence,
            "recommended_action": self.recommended_action.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class GateDecision:
    """Decision from the safety gate."""
    action: SafetyAction
    concerns: List[SafetyConcern]
    confidence: float
    reasoning: str
    modifications: Optional[Dict[str, Any]] = None
    blocked_reason: Optional[str] = None
    processing_time_ms: float = 0.0

    @property
    def should_block(self) -> bool:
        return self.action == SafetyAction.BLOCK

    @property
    def should_modify(self) -> bool:
        return self.action == SafetyAction.MODIFY

    @property
    def has_concerns(self) -> bool:
        return len(self.concerns) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "concerns": [c.to_dict() for c in self.concerns],
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "modifications": self.modifications,
            "blocked_reason": self.blocked_reason,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class SafetyGateConfig:
    """Configuration for the safety gate."""
    # Thresholds for blocking
    block_threshold: float = 0.8            # Block if max severity >= this
    warn_threshold: float = 0.4             # Warn if max severity >= this
    escalate_threshold: float = 0.9         # Escalate if severity >= this

    # Pattern detection thresholds
    deception_threshold: float = 0.6
    power_seeking_threshold: float = 0.5
    hallucination_confidence: float = 0.3   # Block if confidence < this
    drift_threshold: float = 0.7
    anomaly_z_score: float = 3.0            # Z-score threshold for anomalies

    # Feature monitoring
    monitored_features: List[str] = field(default_factory=list)
    blocked_features: List[str] = field(default_factory=list)  # Always block if active

    # Callbacks
    enable_logging: bool = True
    enable_alerts: bool = True
    enable_audit_trail: bool = True


class SafetyPatternDetector:
    """Detects concerning patterns in activations."""

    def __init__(self, config: SafetyGateConfig):
        self.config = config
        self._baseline_activations: Optional[np.ndarray] = None
        self._activation_history: List[np.ndarray] = []

    def set_baseline(self, activations: np.ndarray) -> None:
        """Set baseline activation patterns."""
        self._baseline_activations = activations.copy()

    def add_to_history(self, activations: np.ndarray) -> None:
        """Add activations to history for anomaly detection."""
        self._activation_history.append(activations)
        if len(self._activation_history) > 1000:
            self._activation_history = self._activation_history[-1000:]

    def detect_deception(
        self,
        activations: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Optional[SafetyConcern]:
        """Detect potential deception patterns."""
        # Look for features associated with deception
        deception_features = [
            "dishonesty", "misleading", "false", "deceptive",
            "manipulation", "pretend", "hide", "conceal"
        ]

        if feature_names:
            deception_scores = []
            affected = []
            for i, name in enumerate(feature_names):
                for df in deception_features:
                    if df in name.lower():
                        if i < len(activations):
                            score = float(activations[i]) if activations[i] > 0 else 0.0
                            if score > 0.1:
                                deception_scores.append(score)
                                affected.append(name)

            if deception_scores:
                max_score = max(deception_scores)
                if max_score >= self.config.deception_threshold:
                    return SafetyConcern(
                        concern_type=ConcernType.DECEPTION,
                        severity=max_score,
                        description=f"Deception-associated features detected (score: {max_score:.3f})",
                        affected_features=affected,
                        evidence={"scores": deception_scores},
                        recommended_action=SafetyAction.BLOCK if max_score > 0.8 else SafetyAction.WARN,
                    )

        return None

    def detect_power_seeking(
        self,
        activations: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Optional[SafetyConcern]:
        """Detect power-seeking behavior patterns."""
        power_features = [
            "control", "power", "dominate", "acquire", "resource",
            "self-preservation", "avoid_shutdown", "persist"
        ]

        if feature_names:
            power_scores = []
            affected = []
            for i, name in enumerate(feature_names):
                for pf in power_features:
                    if pf in name.lower():
                        if i < len(activations):
                            score = float(activations[i]) if activations[i] > 0 else 0.0
                            if score > 0.1:
                                power_scores.append(score)
                                affected.append(name)

            if power_scores:
                max_score = max(power_scores)
                if max_score >= self.config.power_seeking_threshold:
                    return SafetyConcern(
                        concern_type=ConcernType.POWER_SEEKING,
                        severity=max_score,
                        description=f"Power-seeking patterns detected (score: {max_score:.3f})",
                        affected_features=affected,
                        evidence={"scores": power_scores},
                        recommended_action=SafetyAction.ESCALATE if max_score > 0.8 else SafetyAction.WARN,
                    )

        return None

    def detect_anomaly(
        self,
        activations: np.ndarray,
    ) -> Optional[SafetyConcern]:
        """Detect anomalous activation patterns."""
        if len(self._activation_history) < 10:
            return None

        # Calculate z-scores
        history_array = np.array(self._activation_history)
        mean = np.mean(history_array, axis=0)
        std = np.std(history_array, axis=0)
        std = np.where(std < 1e-8, 1e-8, std)  # Avoid division by zero

        z_scores = np.abs((activations - mean) / std)
        max_z = float(np.max(z_scores))
        anomaly_indices = np.where(z_scores > self.config.anomaly_z_score)[0]

        if len(anomaly_indices) > 0:
            severity = min(1.0, max_z / 10.0)  # Normalize to 0-1
            return SafetyConcern(
                concern_type=ConcernType.ANOMALY,
                severity=severity,
                description=f"Anomalous activations detected (max z-score: {max_z:.2f})",
                affected_features=[f"feature_{i}" for i in anomaly_indices[:5]],
                evidence={
                    "max_z_score": max_z,
                    "n_anomalies": len(anomaly_indices),
                    "anomaly_indices": anomaly_indices[:10].tolist(),
                },
                recommended_action=SafetyAction.WARN if severity < 0.7 else SafetyAction.BLOCK,
            )

        return None

    def detect_drift(
        self,
        activations: np.ndarray,
    ) -> Optional[SafetyConcern]:
        """Detect significant drift from baseline."""
        if self._baseline_activations is None:
            return None

        # Calculate drift magnitude
        diff = activations - self._baseline_activations
        drift_magnitude = float(np.linalg.norm(diff)) / (np.linalg.norm(self._baseline_activations) + 1e-8)

        if drift_magnitude >= self.config.drift_threshold:
            severity = min(1.0, drift_magnitude)
            return SafetyConcern(
                concern_type=ConcernType.DRIFT,
                severity=severity,
                description=f"Significant drift from baseline (magnitude: {drift_magnitude:.3f})",
                evidence={"drift_magnitude": drift_magnitude},
                recommended_action=SafetyAction.WARN,
            )

        return None


class DarkTraceSafetyGate:
    """
    Automatic safety gate using Dark Trace analysis.

    Analyzes model activations and behavior to detect concerning
    patterns and automatically block or modify responses.
    """

    def __init__(self, config: Optional[SafetyGateConfig] = None):
        self.config = config or SafetyGateConfig()
        self.detector = SafetyPatternDetector(self.config)
        self._decision_history: List[GateDecision] = []
        self._alert_callback: Optional[Callable[[SafetyConcern], Awaitable[None]]] = None
        self._audit_callback: Optional[Callable[[GateDecision], Awaitable[None]]] = None

    def set_baseline(self, activations: np.ndarray) -> None:
        """Set baseline activation patterns for drift detection."""
        self.detector.set_baseline(activations)

    def set_alert_callback(
        self,
        callback: Callable[[SafetyConcern], Awaitable[None]]
    ) -> None:
        """Set callback for safety alerts."""
        self._alert_callback = callback

    def set_audit_callback(
        self,
        callback: Callable[[GateDecision], Awaitable[None]]
    ) -> None:
        """Set callback for audit trail."""
        self._audit_callback = callback

    async def evaluate(
        self,
        activations: np.ndarray,
        confidence: float = 1.0,
        feature_names: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> GateDecision:
        """
        Evaluate activations and make a safety decision.

        Args:
            activations: Model activations to analyze
            confidence: Model confidence in the response
            feature_names: Names of features for pattern detection
            context: Additional context for decision-making

        Returns:
            GateDecision with action and concerns
        """
        import time
        start_time = time.perf_counter()

        concerns: List[SafetyConcern] = []

        # Check for blocked features
        if feature_names and self.config.blocked_features:
            for i, name in enumerate(feature_names):
                if name in self.config.blocked_features:
                    if i < len(activations) and activations[i] > 0.1:
                        concerns.append(SafetyConcern(
                            concern_type=ConcernType.HARMFUL_CONTENT,
                            severity=1.0,
                            description=f"Blocked feature '{name}' is active",
                            affected_features=[name],
                            recommended_action=SafetyAction.BLOCK,
                        ))

        # Low confidence = potential hallucination
        if confidence < self.config.hallucination_confidence:
            concerns.append(SafetyConcern(
                concern_type=ConcernType.HALLUCINATION,
                severity=1.0 - confidence,
                description=f"Low confidence ({confidence:.3f}) indicates potential hallucination",
                evidence={"confidence": confidence},
                recommended_action=SafetyAction.WARN if confidence > 0.2 else SafetyAction.BLOCK,
            ))

        # Detect patterns
        deception = self.detector.detect_deception(activations, feature_names)
        if deception:
            concerns.append(deception)

        power_seeking = self.detector.detect_power_seeking(activations, feature_names)
        if power_seeking:
            concerns.append(power_seeking)

        anomaly = self.detector.detect_anomaly(activations)
        if anomaly:
            concerns.append(anomaly)

        drift = self.detector.detect_drift(activations)
        if drift:
            concerns.append(drift)

        # Add to history for future anomaly detection
        self.detector.add_to_history(activations)

        # Determine action
        action = SafetyAction.ALLOW
        reasoning = "No concerns detected"
        blocked_reason = None

        if concerns:
            max_severity = max(c.severity for c in concerns)
            critical_concerns = [c for c in concerns if c.severity >= self.config.block_threshold]
            escalation_concerns = [c for c in concerns if c.severity >= self.config.escalate_threshold]

            if escalation_concerns:
                action = SafetyAction.ESCALATE
                reasoning = f"Escalating due to {len(escalation_concerns)} critical concern(s)"
            elif critical_concerns:
                action = SafetyAction.BLOCK
                blocked_reason = critical_concerns[0].description
                reasoning = f"Blocking due to {len(critical_concerns)} high-severity concern(s)"
            elif max_severity >= self.config.warn_threshold:
                action = SafetyAction.WARN
                reasoning = f"Warning: {len(concerns)} concern(s) detected (max severity: {max_severity:.2f})"
            else:
                reasoning = f"Low-severity concerns detected but below threshold"

        processing_time = (time.perf_counter() - start_time) * 1000

        decision = GateDecision(
            action=action,
            concerns=concerns,
            confidence=1.0 - (max(c.severity for c in concerns) if concerns else 0.0),
            reasoning=reasoning,
            blocked_reason=blocked_reason,
            processing_time_ms=processing_time,
        )

        # Fire callbacks
        if self.config.enable_alerts and self._alert_callback:
            for concern in concerns:
                if concern.severity >= self.config.warn_threshold:
                    await self._alert_callback(concern)

        if self.config.enable_audit_trail and self._audit_callback:
            await self._audit_callback(decision)

        # Store in history
        self._decision_history.append(decision)
        if len(self._decision_history) > 1000:
            self._decision_history = self._decision_history[-1000:]

        return decision

    def get_statistics(self) -> Dict[str, Any]:
        """Get gate statistics."""
        if not self._decision_history:
            return {"total_evaluations": 0}

        total = len(self._decision_history)
        actions = {}
        for d in self._decision_history:
            action = d.action.value
            actions[action] = actions.get(action, 0) + 1

        concern_types = {}
        for d in self._decision_history:
            for c in d.concerns:
                ct = c.concern_type.value
                concern_types[ct] = concern_types.get(ct, 0) + 1

        avg_processing = sum(d.processing_time_ms for d in self._decision_history) / total

        return {
            "total_evaluations": total,
            "actions": actions,
            "concern_types": concern_types,
            "block_rate": actions.get("block", 0) / total,
            "warn_rate": actions.get("warn", 0) / total,
            "avg_processing_ms": avg_processing,
        }


async def create_safety_gate(
    config: Optional[SafetyGateConfig] = None,
    baseline_activations: Optional[np.ndarray] = None,
) -> DarkTraceSafetyGate:
    """
    Create and configure a safety gate.

    Args:
        config: Gate configuration
        baseline_activations: Optional baseline for drift detection

    Returns:
        Configured DarkTraceSafetyGate
    """
    gate = DarkTraceSafetyGate(config)

    if baseline_activations is not None:
        gate.set_baseline(baseline_activations)

    return gate
