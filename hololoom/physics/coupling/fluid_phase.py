"""
Fluid Phase — knowledge propagation via pressure gradients.

Reads: routing_target, routing_confidence (from Gradient)
Writes: pressure, per-node pressures

The routing target gets pressure injection. High-pressure (important)
flows to low-pressure (sparse) regions via Navier-Stokes.
"""
from __future__ import annotations

import logging

import numpy as np

from .protocol import PhysicsPhase, PhysicsState

logger = logging.getLogger(__name__)

try:
    from hololoom.physics.fluid_dynamics import ContextFlowEngine
    FLUID_AVAILABLE = True
except ImportError:
    FLUID_AVAILABLE = False
    logger.debug("ContextFlowEngine unavailable")


class FluidPhase:
    """Adapts ContextFlowEngine into the coupled ring."""

    name = "fluid"

    def __init__(self, viscosity: float = 0.1):
        self._viscosity = viscosity
        self._engine = None
        if FLUID_AVAILABLE:
            try:
                self._engine = ContextFlowEngine(viscosity=viscosity)
            except Exception:
                pass

    def available(self) -> bool:
        return FLUID_AVAILABLE and self._engine is not None

    def step(self, state: PhysicsState) -> PhysicsState:
        """Propagate fluid flow. Inject pressure at routing target."""
        # Simple pressure propagation without full engine
        # (engine requires graph topology we may not have)
        pressures = state.pressures

        # Inject pressure at routing target
        if state.routing_target and state.routing_confidence > 0:
            current = pressures.get(state.routing_target, 0.0)
            pressures[state.routing_target] = current + state.routing_confidence * 0.5

        # Diffuse pressure: each node moves toward mean
        if pressures:
            mean_p = np.mean(list(pressures.values()))
            for node in list(pressures.keys()):
                p = pressures[node]
                # Viscous diffusion toward mean
                dp = -self._viscosity * (p - mean_p) * state.dt
                pressures[node] = max(0.0, p + dp)

            # Remove negligible pressures
            state.pressures = {k: v for k, v in pressures.items() if v > 1e-4}

            # Compute scalar: pressure variance (drives energy)
            if state.pressures:
                vals = list(state.pressures.values())
                state.pressure = float(np.mean(vals))
            else:
                state.pressure = 0.0
        else:
            state.pressure = 0.0

        return state
