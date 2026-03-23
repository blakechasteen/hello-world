"""
Hierarchical Wave Propagation — same wave equation at every scale.

Four levels: sub_note (fast, c=2.0) -> note -> department -> cross_department (slow, c=0.2).
When constructive interference exceeds threshold at a boundary,
it composes up to the next level. At cross_department, boundary
events become PROACTIVE agentic intents.

Waves as agentic calls: a wave IS a message carrying amplitude (importance),
frequency (change rate), and phase (timing). This isn't metaphor — it's the mechanism.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .protocol import PhysicsState

logger = logging.getLogger(__name__)

try:
    from hololoom.physics.wave_mechanics import WaveMechanicsEngine
    WAVE_ENGINE_AVAILABLE = True
except ImportError:
    WAVE_ENGINE_AVAILABLE = False


LEVEL_ORDER = ("sub_note", "note", "department", "cross_department")

LEVEL_CONFIG = {
    "sub_note":         {"wave_speed": 2.0, "damping": 0.10},
    "note":             {"wave_speed": 1.0, "damping": 0.08},
    "department":       {"wave_speed": 0.5, "damping": 0.15},
    "cross_department": {"wave_speed": 0.2, "damping": 0.20},
}


@dataclass
class BoundaryEvent:
    """A wave crossed a level boundary."""
    source_level: str
    target_level: str
    amplitude: float
    nodes: list[str] = field(default_factory=list)
    action: str = "amplify"   # "amplify" (constructive) or "dampen" (destructive)
    strength: float = 0.0


class WaveLevel:
    """Wave propagation at a single hierarchical level."""

    def __init__(self, name: str, wave_speed: float, damping: float):
        self.name = name
        self.wave_speed = wave_speed
        self.damping = damping
        self.amplitudes: dict[str, float] = {}
        self.velocities: dict[str, float] = {}

    def inject(self, node: str, amplitude: float) -> None:
        """Inject a wave at a node."""
        current = self.amplitudes.get(node, 0.0)
        self.amplitudes[node] = current + amplitude

    def step(self, dt: float = 0.01) -> None:
        """Propagate waves at this level."""
        if not self.amplitudes:
            return

        mean_amp = np.mean(list(self.amplitudes.values()))

        for node in list(self.amplitudes.keys()):
            a = self.amplitudes[node]
            v = self.velocities.get(node, 0.0)

            # Wave equation: acceleration = c^2 * laplacian - damping * velocity
            # Mean-field laplacian: (mean - a)
            accel = self.wave_speed ** 2 * (mean_amp - a) - self.damping * v
            v_new = v + accel * dt
            a_new = a + v_new * dt

            self.amplitudes[node] = a_new
            self.velocities[node] = v_new

        # Clean negligible
        self.amplitudes = {k: v for k, v in self.amplitudes.items() if abs(v) > 1e-5}
        self.velocities = {k: v for k, v in self.velocities.items()
                          if k in self.amplitudes}

    def detect_boundary_events(self, threshold: float = 0.5) -> list[tuple[str, float, str]]:
        """Detect nodes with amplitude exceeding threshold.

        Returns list of (node, amplitude, interference_type).
        """
        events = []
        for node, amp in self.amplitudes.items():
            if abs(amp) > threshold:
                itype = "constructive" if amp > 0 else "destructive"
                events.append((node, amp, itype))
        return events

    @property
    def total_energy(self) -> float:
        return sum(a ** 2 for a in self.amplitudes.values())


class WaveHierarchy:
    """Same wave equation at every scale. Boundary crossings compose upward."""

    def __init__(self, boundary_threshold: float = 0.5):
        self._levels: dict[str, WaveLevel] = {}
        for name, config in LEVEL_CONFIG.items():
            self._levels[name] = WaveLevel(name, **config)
        self._threshold = boundary_threshold

    def inject(self, level: str, node: str, amplitude: float) -> None:
        """Inject a wave at a specific level and node."""
        if level in self._levels:
            self._levels[level].inject(node, amplitude)

    def propagate(self, dt: float = 0.01) -> list[BoundaryEvent]:
        """Propagate all levels. Detect and handle boundary crossings."""
        boundary_events: list[BoundaryEvent] = []

        for i, level_name in enumerate(LEVEL_ORDER):
            level = self._levels[level_name]
            level.step(dt)

            # Check for boundary crossings
            events = level.detect_boundary_events(self._threshold)
            if not events:
                continue

            # Get next level (if exists)
            next_idx = i + 1
            if next_idx >= len(LEVEL_ORDER):
                # Top level — these become agentic intents
                for node, amp, itype in events:
                    boundary_events.append(BoundaryEvent(
                        source_level=level_name,
                        target_level="intent",
                        amplitude=amp,
                        nodes=[node],
                        action="amplify" if itype == "constructive" else "dampen",
                        strength=abs(amp),
                    ))
                continue

            next_name = LEVEL_ORDER[next_idx]

            for node, amp, itype in events:
                if itype == "constructive":
                    # Amplify at next scale (attenuate to prevent runaway)
                    compose_amp = min(amp * 0.3, 2.0)
                    self._levels[next_name].inject(node, compose_amp)
                    boundary_events.append(BoundaryEvent(
                        source_level=level_name,
                        target_level=next_name,
                        amplitude=compose_amp,
                        nodes=[node],
                        action="amplify",
                        strength=abs(amp),
                    ))
                else:
                    # Dampen at next scale
                    self._levels[next_name].inject(node, min(amp * 0.2, 1.0))
                    boundary_events.append(BoundaryEvent(
                        source_level=level_name,
                        target_level=next_name,
                        amplitude=amp * 0.3,
                        nodes=[node],
                        action="dampen",
                        strength=abs(amp),
                    ))

        return boundary_events

    @staticmethod
    def boundary_to_intent(event: BoundaryEvent) -> dict:
        """Convert a boundary crossing into an agentic intent dict.

        At cross_department level, waves become PROACTIVE intents.
        """
        return {
            "origin": "PROACTIVE",
            "source": f"wave_boundary:{event.source_level}",
            "action": event.action,
            "nodes": event.nodes,
            "strength": round(event.strength, 3),
            "amplitude": round(event.amplitude, 3),
        }

    def inject_from_state(self, state: PhysicsState) -> None:
        """Seed waves from physics state activations."""
        for node, activation in state.activations.items():
            if abs(activation) > 0.1:
                self._levels["note"].inject(node, activation * 0.1)

    def status(self) -> dict:
        """Summary for visualization."""
        return {
            level_name: {
                "n_active": len(level.amplitudes),
                "total_energy": round(level.total_energy, 4),
            }
            for level_name, level in self._levels.items()
        }
