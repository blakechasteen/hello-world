"""
Thermo Phase — exploration/exploitation via free energy minimization.

Reads: pressure (from Fluid — pressure variance -> energy)
Writes: temperature, entropy, energy, free_energy

Helmholtz: F = E - TS.
High T -> entropy dominates -> exploration.
Low T -> energy dominates -> exploitation.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from .protocol import PhysicsPhase, PhysicsState

logger = logging.getLogger(__name__)

try:
    from hololoom.physics.thermodynamics import (
        ThermodynamicOptimizer,
        EnergyCalculator,
        EntropyCalculator,
    )
    THERMO_AVAILABLE = True
except ImportError:
    THERMO_AVAILABLE = False
    logger.debug("ThermodynamicOptimizer unavailable")


class ThermoPhase:
    """Adapts ThermodynamicOptimizer into the coupled ring."""

    name = "thermo"

    def available(self) -> bool:
        return True  # Core math only, no optional deps

    def step(self, state: PhysicsState) -> PhysicsState:
        """Compute thermodynamic quantities from pressure distribution."""
        # Energy from pressure: variance in pressure = internal energy
        if state.pressures:
            vals = list(state.pressures.values())
            pressure_var = float(np.var(vals)) if len(vals) > 1 else 0.0
            state.energy = state.tension + pressure_var
        else:
            state.energy = state.tension

        # Entropy from activation distribution
        if state.activations:
            vals = np.array(list(state.activations.values()))
            vals = np.abs(vals)
            total = vals.sum()
            if total > 0:
                probs = vals / total
                # Shannon entropy (normalized)
                entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
                max_entropy = math.log(max(len(probs), 2))
                state.entropy = entropy / max(max_entropy, 1.0)
            else:
                state.entropy = 1.0  # Maximum entropy = no information
        else:
            state.entropy = 1.0

        # Free energy: F = E - TS
        state.free_energy = state.energy - state.temperature * state.entropy

        return state
