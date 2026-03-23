"""
StatMech Phase — emergent behavior via Boltzmann distribution.

Reads: amplitudes (from Wave — interference patterns become microstates)
Writes: order_parameter, phase_type, entropy contribution

Constructive interference clusters = low-energy microstates.
Destructive interference = high-energy microstates.
Partition function normalizes. Phase transitions detected via order parameter.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from .protocol import PhysicsPhase, PhysicsState

logger = logging.getLogger(__name__)

try:
    from hololoom.physics.statistical_mechanics import (
        StatisticalMechanicsEngine,
        Microstate,
        CanonicalEnsemble,
    )
    STATMECH_AVAILABLE = True
except ImportError:
    STATMECH_AVAILABLE = False
    logger.debug("StatisticalMechanicsEngine unavailable")


class StatMechPhase:
    """Adapts StatisticalMechanicsEngine into the coupled ring."""

    name = "statmech"

    def __init__(self, critical_temperature: float = 0.5):
        self._T_c = critical_temperature

    def available(self) -> bool:
        return True  # Core math only

    def step(self, state: PhysicsState) -> PhysicsState:
        """Compute ensemble averages from wave amplitudes."""
        if not state.amplitudes:
            return state

        T = max(state.temperature, 0.01)  # Avoid division by zero
        amplitudes = list(state.amplitudes.values())

        # Treat amplitudes as microstate energies
        # Constructive interference (high amplitude) = low energy
        # Destructive interference (low amplitude) = high energy
        energies = np.array([1.0 / (a + 0.01) for a in amplitudes])

        # Boltzmann weights: w_i = exp(-E_i / T)
        boltzmann = np.exp(-energies / T)
        Z = boltzmann.sum()  # Partition function

        if Z > 0:
            probabilities = boltzmann / Z

            # Average energy
            avg_energy = float(np.sum(probabilities * energies))

            # Gibbs entropy: S = -Σ p_i ln(p_i)
            gibbs_entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-10)))
            max_entropy = math.log(max(len(probabilities), 2))
            normalized_entropy = gibbs_entropy / max(max_entropy, 1.0)

            # Order parameter: 1 - normalized entropy
            # High order = low entropy = organized
            # Low order = high entropy = disordered
            order = max(0.0, min(1.0, 1.0 - normalized_entropy))

            # Blend with current (smooth transitions)
            state.order_parameter = 0.8 * state.order_parameter + 0.2 * order

            # Blend entropy
            state.entropy = 0.7 * state.entropy + 0.3 * normalized_entropy

        # Phase classification
        if state.order_parameter > 0.7:
            state.phase_type = "ordered"
        elif state.order_parameter > 0.3:
            state.phase_type = "critical"
        else:
            state.phase_type = "disordered"

        return state
