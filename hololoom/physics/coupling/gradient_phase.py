"""
Gradient Phase — queries flow downhill through loss landscapes.

Reads: tension (from Spring — tensions become loss features)
Writes: gradient_loss, routing_target, routing_confidence

Spring tension feeds the loss landscape: high tension = high loss for
that path. The gradient engine finds the lowest-loss route.
"""
from __future__ import annotations

import logging

import numpy as np

from .protocol import PhysicsPhase, PhysicsState

logger = logging.getLogger(__name__)

try:
    from hololoom.physics.gradient_flow import GradientFlowEngine
    GRADIENT_AVAILABLE = True
except ImportError:
    GRADIENT_AVAILABLE = False
    logger.debug("GradientFlowEngine unavailable")


def _default_loss(metrics: dict[str, float]) -> float:
    """Combined loss: weighted sum of cost and tension."""
    return 0.5 * metrics.get("cost", 0.5) + 0.5 * metrics.get("tension", 0.5)


class GradientPhase:
    """Adapts GradientFlowEngine into the coupled ring."""

    name = "gradient"

    def __init__(self, learning_rate: float = 0.1, noise_level: float = 0.05):
        self._lr = learning_rate
        self._noise = noise_level
        if GRADIENT_AVAILABLE:
            self._engine = GradientFlowEngine(
                loss_fn=_default_loss,
                learning_rate=learning_rate,
                noise_level=noise_level,
            )
        else:
            self._engine = None

    def available(self) -> bool:
        return GRADIENT_AVAILABLE and self._engine is not None

    def step(self, state: PhysicsState) -> PhysicsState:
        """Route using gradient flow. Tension feeds loss landscape."""
        if not self._engine:
            return state

        # Build targets from active nodes
        # Each node's tension contributes to its routing cost
        nodes = list(state.activations.keys())
        if len(nodes) < 2:
            state.gradient_loss = state.tension
            return state

        targets = []
        target_metrics = []
        for node in nodes[:20]:  # Cap at 20 candidates
            tension_val = abs(state.activations.get(node, 0.0))
            targets.append(node)
            target_metrics.append({
                "cost": tension_val,  # High activation = high cost
                "tension": state.tension,
            })

        try:
            decision = self._engine.route(
                targets=targets,
                target_metrics=target_metrics,
                max_steps=5,
            )
            state.gradient_loss = decision.loss
            state.routing_target = decision.target
            state.routing_confidence = max(0.0, 1.0 - decision.loss)
        except Exception as e:
            logger.debug(f"Gradient routing failed: {e}")
            state.gradient_loss = state.tension

        return state
