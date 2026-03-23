"""
Wave Phase — pattern detection via interference.

Reads: temperature (from Thermo — sets damping), free_energy (modulates amplitude)
Writes: wave_energy, per-node amplitudes

Temperature controls damping: high T = more damping (energy dissipates faster).
Free energy modulates wave injection amplitude.
"""
from __future__ import annotations

import logging

import numpy as np

from .protocol import PhysicsPhase, PhysicsState

logger = logging.getLogger(__name__)

try:
    from hololoom.physics.wave_mechanics import WaveMechanicsEngine
    WAVE_AVAILABLE = True
except ImportError:
    WAVE_AVAILABLE = False
    logger.debug("WaveMechanicsEngine unavailable")


class WavePhase:
    """Adapts WaveMechanicsEngine into the coupled ring."""

    name = "wave"

    def __init__(self, base_speed: float = 1.0, base_damping: float = 0.01):
        self._base_speed = base_speed
        self._base_damping = base_damping

    def available(self) -> bool:
        return True  # Uses inline wave equation if engine unavailable

    def step(self, state: PhysicsState) -> PhysicsState:
        """Propagate waves. Temperature sets damping."""
        # Damping = base * (1 + temperature) — hotter = more damping
        damping = self._base_damping * (1.0 + state.temperature)

        # Injection amplitude scales inversely with free energy
        # Low free energy = strong waves (system is organized)
        inject_amp = max(0.01, 1.0 / (1.0 + abs(state.free_energy)))

        amplitudes = state.amplitudes

        # Inject waves at nodes with high activation
        for node, activation in state.activations.items():
            if abs(activation) > 0.1:
                current = amplitudes.get(node, 0.0)
                amplitudes[node] = current + inject_amp * abs(activation) * state.dt

        # Propagate: diffuse amplitudes between neighbors (mean field)
        if amplitudes:
            mean_amp = np.mean(list(amplitudes.values()))
            total_energy = 0.0

            for node in list(amplitudes.keys()):
                a = amplitudes[node]
                # Diffusion + damping
                da = -damping * a + 0.1 * (mean_amp - a)
                a_new = a + da * state.dt
                amplitudes[node] = max(0.0, a_new)
                total_energy += a_new ** 2

            # Clean negligible
            state.amplitudes = {k: v for k, v in amplitudes.items() if v > 1e-4}
            state.wave_energy = total_energy
        else:
            state.wave_energy = 0.0

        return state
