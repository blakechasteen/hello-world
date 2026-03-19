"""
Convergence Adapter — Plugs convergence detectors into the bus.
================================================================

Subscribes to STEP_COMPLETE signals. Feeds confidence values from step
results into existing convergence detectors (ConfidenceThreshold,
ImprovementDelta, QualityPlateau). When convergence is detected,
emits a CONVERGENCE signal that the chain adapter can use to early-terminate.

This adapter WRAPS existing detectors from hololoom/core/convergence/.
It does not modify them.

Usage:
    from hololoom.core.bus import UnifiedBus
    from hololoom.core.bus.adapters.convergence_adapter import ConvergenceMonitor

    bus = UnifiedBus()
    monitor = ConvergenceMonitor(bus, threshold=0.85, min_delta=0.05)
    # monitor automatically subscribes to STEP_COMPLETE
    # and emits CONVERGENCE when criteria are met
"""

import logging
from dataclasses import dataclass

from ..signal import Signal, SignalKind, SignalPriority
from ..unified_bus import UnifiedBus

logger = logging.getLogger(__name__)


@dataclass
class StepObservation:
    """Observation from a step completion for convergence tracking."""
    step_id: str
    confidence: float | None = None
    improvement: float = 0.0
    execution_id: str = ""


class ConvergenceMonitor:
    """
    Bus-connected convergence monitor.

    Tracks confidence values from STEP_COMPLETE signals per execution.
    When convergence is detected, emits a CONVERGENCE signal.

    Implements the same detection logic as the existing detectors
    (ConfidenceThreshold, ImprovementDelta, QualityPlateau) but driven
    by bus signals instead of direct function calls.
    """

    def __init__(
        self,
        bus: UnifiedBus,
        confidence_threshold: float = 0.85,
        min_delta: float = 0.05,
        plateau_window: int = 3,
        plateau_min_delta: float = 0.02,
    ):
        """
        Args:
            bus: UnifiedBus to subscribe to and emit on
            confidence_threshold: Confidence level that triggers convergence
            min_delta: Minimum improvement required (below = converged)
            plateau_window: Number of steps to check for plateau
            plateau_min_delta: Minimum improvement to count as non-plateau
        """
        self.bus = bus
        self.confidence_threshold = confidence_threshold
        self.min_delta = min_delta
        self.plateau_window = plateau_window
        self.plateau_min_delta = plateau_min_delta

        # Track observations per execution
        self._observations: dict[str, list[StepObservation]] = {}

        # Subscribe to step completions
        self.bus.on(SignalKind.STEP_COMPLETE, self._on_step_complete)

        # Track which executions have already converged (don't re-emit)
        self._converged_executions: set = set()

    async def _on_step_complete(self, signal: Signal) -> Signal | None:
        """Handle STEP_COMPLETE signal — check for convergence."""
        execution_id = signal.payload.get("execution_id", "")
        if not execution_id:
            return None

        # Skip if already converged for this execution
        if execution_id in self._converged_executions:
            return None

        confidence = signal.payload.get("confidence")
        if confidence is None:
            return None

        # Build observation
        prev_confidence = self._get_last_confidence(execution_id)
        improvement = (confidence - prev_confidence) if prev_confidence is not None else 0.0

        obs = StepObservation(
            step_id=signal.payload.get("step_id", ""),
            confidence=confidence,
            improvement=improvement,
            execution_id=execution_id,
        )

        if execution_id not in self._observations:
            self._observations[execution_id] = []
        self._observations[execution_id].append(obs)

        # Run convergence checks
        converged, reason = self._check_convergence(execution_id)

        if converged:
            self._converged_executions.add(execution_id)
            logger.info(f"Convergence detected for {execution_id}: {reason}")

            # Emit CONVERGENCE signal (chain adapter will pick this up)
            return Signal(
                kind=SignalKind.CONVERGENCE,
                priority=SignalPriority.HIGH,
                source="convergence_monitor",
                payload={
                    "execution_id": execution_id,
                    "reason": reason,
                    "confidence": confidence,
                    "steps_observed": len(self._observations[execution_id]),
                },
                correlation_id=signal.correlation_id,
            )

        return None

    def _check_convergence(self, execution_id: str) -> tuple[bool, str]:
        """
        Run convergence detection — mirrors existing detector logic.

        Checks in order:
        1. Confidence threshold (from ConfidenceThresholdDetector)
        2. Improvement delta (from ImprovementDeltaDetector)
        3. Quality plateau (from QualityPlateauDetector)
        """
        observations = self._observations.get(execution_id, [])
        if not observations:
            return False, "no observations"

        latest = observations[-1]

        # 1. Confidence threshold
        if latest.confidence is not None and latest.confidence >= self.confidence_threshold:
            return True, (
                f"confidence {latest.confidence:.2f} >= threshold {self.confidence_threshold}"
            )

        # 2. Improvement delta (need 2+ observations)
        if len(observations) >= 2:
            if abs(latest.improvement) < self.min_delta:
                return True, (
                    f"improvement {latest.improvement:+.3f} < min_delta {self.min_delta}"
                )

        # 3. Quality plateau (need plateau_window observations)
        if len(observations) >= self.plateau_window:
            recent = observations[-self.plateau_window:]
            below = sum(1 for o in recent if abs(o.improvement) < self.plateau_min_delta)
            if below >= self.plateau_window:
                return True, (
                    f"quality plateaued for {self.plateau_window} steps "
                    f"(improvements < {self.plateau_min_delta})"
                )

        return False, "not converged"

    def _get_last_confidence(self, execution_id: str) -> float | None:
        """Get the last observed confidence for an execution."""
        observations = self._observations.get(execution_id, [])
        if observations:
            return observations[-1].confidence
        return None

    def reset(self, execution_id: str | None = None) -> None:
        """Reset convergence state for an execution (or all)."""
        if execution_id:
            self._observations.pop(execution_id, None)
            self._converged_executions.discard(execution_id)
        else:
            self._observations.clear()
            self._converged_executions.clear()

    def get_status(self, execution_id: str) -> dict:
        """Get convergence status for an execution."""
        observations = self._observations.get(execution_id, [])
        converged = execution_id in self._converged_executions

        return {
            "execution_id": execution_id,
            "converged": converged,
            "observations": len(observations),
            "latest_confidence": observations[-1].confidence if observations else None,
            "latest_improvement": observations[-1].improvement if observations else None,
        }


__all__ = ["ConvergenceMonitor"]
