"""
Spring Phase — Hooke's Law spreading activation.

Reads: activations, velocities, order_parameter (from StatMech)
Writes: tension, updated activations/velocities

The order parameter modulates spring damping: high order = more damping
(system settling), low order = less damping (more exploration).
"""
from __future__ import annotations

import logging

import numpy as np

from .protocol import PhysicsPhase, PhysicsState

logger = logging.getLogger(__name__)

# Try to import the engine
try:
    from hololoom.core.memory.spring_dynamics import SpringConfig
    SPRING_AVAILABLE = True
except ImportError:
    SPRING_AVAILABLE = False
    logger.debug("SpringDynamics unavailable")


class SpringPhase:
    """Adapts SpringDynamics into the coupled physics ring."""

    name = "spring"

    def __init__(self, stiffness: float = 0.5, base_damping: float = 0.1):
        self._stiffness = stiffness
        self._base_damping = base_damping

    def available(self) -> bool:
        return SPRING_AVAILABLE

    def step(self, state: PhysicsState) -> PhysicsState:
        """Propagate spring dynamics. Order parameter modulates damping."""
        if not state.activations:
            return state

        # Damping increases with order (system settling into structure)
        damping = self._base_damping * (1.0 + state.order_parameter)

        # Hooke's Law: F = -k * x - c * v
        # Simplified for state dict (no graph topology — uses mean field)
        nodes = list(state.activations.keys())
        if not nodes:
            return state

        activations = state.activations
        velocities = state.velocities

        mean_activation = np.mean(list(activations.values())) if activations else 0.0
        total_tension = 0.0

        for node in nodes:
            a = activations.get(node, 0.0)
            v = velocities.get(node, 0.0)

            # Spring force toward mean (coupling)
            displacement = a - mean_activation
            force = -self._stiffness * displacement - damping * v

            # Velocity Verlet integration
            v_new = v + force * state.dt
            a_new = a + v_new * state.dt

            # Decay activations that are too small
            if abs(a_new) < 1e-4:
                a_new = 0.0
                v_new = 0.0

            activations[node] = a_new
            velocities[node] = v_new
            total_tension += abs(displacement) * self._stiffness

        # Clean zero entries
        state.activations = {k: v for k, v in activations.items() if v != 0.0}
        state.velocities = {k: v for k, v in velocities.items()
                           if k in state.activations}

        # Write tension scalar
        state.tension = total_tension / max(len(nodes), 1)

        return state
