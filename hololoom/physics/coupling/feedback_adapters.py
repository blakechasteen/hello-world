"""
Physics Feedback Adapters — physics state enters the feedback loop.

PhysicsSourceAdapter: emits physics state as FeedbackSignals
PhysicsConsumerAdapter: receives outcome signals, adjusts coupler

The loop closes: physics computes state -> signals emitted ->
FeedbackHub fuses -> consumers adjust physics parameters ->
next physics step adapts.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .protocol import PhysicsState

logger = logging.getLogger(__name__)

# Try to import feedback protocols
try:
    from hololoom.core.protocols.feedback import (
        FeedbackSignal,
        Timescale,
    )
    FEEDBACK_AVAILABLE = True
except ImportError:
    FEEDBACK_AVAILABLE = False
    logger.debug("FeedbackHub protocols unavailable")

if TYPE_CHECKING:
    from .coupler import PhysicsCoupler


class PhysicsSourceAdapter:
    """Emits physics state as FeedbackSignals.

    Implements FeedbackSource protocol. Registered with FeedbackHub
    to emit energy, entropy, and phase transition signals.
    """

    source_id = "physics_coupler"
    timescale = Timescale.SUB_SECOND if FEEDBACK_AVAILABLE else None

    def __init__(self, coupler: PhysicsCoupler):
        self._coupler = coupler
        self._prev_state: PhysicsState | None = None

    def collect(self) -> list:
        """Collect physics state signals for FeedbackHub."""
        if not FEEDBACK_AVAILABLE:
            return []

        state = self._coupler.state
        signals = []

        # Energy signal: normalized free energy
        # Negative free energy = good (system minimizing)
        energy_val = max(-1.0, min(1.0, -state.free_energy))
        signals.append(FeedbackSignal(
            source=self.source_id,
            timescale=Timescale.SUB_SECOND,
            subject="physics_system",
            dimension="energy",
            value=energy_val,
            confidence=0.8,
            metadata={
                "T": state.temperature,
                "S": state.entropy,
                "F": state.free_energy,
                "order": state.order_parameter,
                "phase": state.phase_type,
            },
        ))

        # Phase transition detection
        if self._prev_state:
            order_delta = abs(state.order_parameter - self._prev_state.order_parameter)
            if order_delta > 0.1:
                signals.append(FeedbackSignal(
                    source=self.source_id,
                    timescale=Timescale.SECONDS,
                    subject="physics_system",
                    dimension="phase_transition",
                    value=order_delta,
                    confidence=0.9,
                    metadata={
                        "from_phase": self._prev_state.phase_type,
                        "to_phase": state.phase_type,
                    },
                ))

            # Energy trend signal
            energy_delta = state.free_energy - self._prev_state.free_energy
            if abs(energy_delta) > 0.05:
                signals.append(FeedbackSignal(
                    source=self.source_id,
                    timescale=Timescale.SUB_SECOND,
                    subject="physics_system",
                    dimension="energy_trend",
                    value=max(-1.0, min(1.0, -energy_delta)),  # Negative delta = good
                    confidence=0.7,
                ))

        self._prev_state = state.copy()
        return signals


class PhysicsConsumerAdapter:
    """Receives outcome signals, adjusts coupler parameters.

    Implements FeedbackConsumer protocol. When quality is bad,
    increases temperature (more exploration). When convergence
    is good, decreases temperature (more exploitation).
    """

    consumer_id = "physics_consumer"

    def __init__(self, coupler: PhysicsCoupler):
        self._coupler = coupler

    def accepts(self, signal) -> bool:
        """Accept quality, convergence, and outcome credit signals."""
        if not FEEDBACK_AVAILABLE:
            return False
        dim = getattr(signal, "dimension", "")
        return dim in ("quality", "convergence", "outcome_credit", "meta_temperature")

    def apply_feedback(self, signals: list) -> None:
        """Adjust physics parameters based on feedback signals."""
        if not signals:
            return

        state = self._coupler.state

        for sig in signals:
            dim = getattr(sig, "dimension", "")
            val = getattr(sig, "value", 0.0)

            if dim == "quality" and val < -0.5:
                # Bad quality -> increase temperature (more exploration)
                state.temperature = min(5.0, state.temperature * 1.1)
                logger.debug(f"Physics: bad quality ({val:.2f}), T -> {state.temperature:.3f}")

            elif dim == "convergence" and val > 0.5:
                # Good convergence -> decrease temperature (exploit)
                state.temperature = max(0.01, state.temperature * 0.95)
                logger.debug(f"Physics: good convergence ({val:.2f}), T -> {state.temperature:.3f}")

            elif dim == "outcome_credit" and val > 0:
                # Positive credit -> reward current physics state
                # Increase order parameter slightly (reinforce structure)
                state.order_parameter = min(1.0, state.order_parameter + val * 0.05)
