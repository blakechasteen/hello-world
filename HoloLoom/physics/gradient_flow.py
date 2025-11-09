"""
Gradient Flow - Natural Downhill Routing

Queries flow "downhill" through loss landscapes to optimal targets,
like water flowing down a mountain to the lowest valley.

Physics Model:
    Gradient Descent with Noise:
        dθ/dt = -∇L(θ) + η·ξ(t)

    Where:
        θ = state (position in loss landscape)
        L(θ) = loss function (server load, cost, etc.)
        ∇L(θ) = gradient (direction of steepest ascent)
        -∇L(θ) = downhill direction
        η = noise amplitude (exploration)
        ξ(t) = Gaussian noise

Core Insight:
    No central coordinator needed - queries naturally flow to optimal
    targets by following gradients, just like water flows downhill.

Applications:
    - Query routing to least-loaded servers
    - Resource allocation across components
    - Multi-tool selection
    - Load balancing
"""

from typing import Dict, List, Callable, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import random


@dataclass
class Target:
    """A routing target (server, tool, component, etc.)."""
    name: str
    metrics: Dict[str, float]  # load, latency, cost, etc.


@dataclass
class RouteDecision:
    """Result of gradient flow routing."""
    target: str
    loss: float
    gradient: float
    position: float
    iterations: int


class GradientFlowEngine:
    """
    Core gradient flow engine.

    Routes queries by following gradients downhill through loss landscapes.

    Usage:
        engine = GradientFlowEngine(
            loss_fn=combined_loss,
            learning_rate=0.1,
            noise_level=0.05
        )

        # Route query
        decision = engine.route(
            targets=["server1", "server2", "server3"],
            target_metrics=[
                {"load": 0.9, "latency": 50},
                {"load": 0.3, "latency": 30},
                {"load": 0.6, "latency": 40}
            ],
            max_steps=10
        )

        print(f"Route to: {decision.target}")
    """

    def __init__(
        self,
        loss_fn: Callable[[Dict[str, float]], float],
        learning_rate: float = 0.1,
        noise_level: float = 0.05,
        dt: float = 0.01
    ):
        """
        Initialize gradient flow engine.

        Args:
            loss_fn: Function that computes loss from target metrics
            learning_rate: How fast to flow downhill (0.0 - 1.0)
            noise_level: Exploration noise amplitude
            dt: Timestep for integration
        """
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.noise_level = noise_level
        self.dt = dt

    def compute_loss(self, target_metrics: Dict[str, float]) -> float:
        """Compute loss for a target."""
        return self.loss_fn(target_metrics)

    def compute_gradient(
        self,
        targets: List[Target],
        position: float
    ) -> float:
        """
        Compute gradient at current position.

        Uses finite differences:
            gradient ≈ (L(x+ε) - L(x-ε)) / (2ε)

        Args:
            targets: List of routing targets
            position: Current position in target space (0.0 - len(targets)-1)

        Returns:
            Gradient value (direction of steepest ascent)
        """
        epsilon = 0.1

        # Interpolate loss at position + epsilon
        pos_plus = min(position + epsilon, len(targets) - 1)
        loss_plus = self._interpolate_loss(targets, pos_plus)

        # Interpolate loss at position - epsilon
        pos_minus = max(position - epsilon, 0)
        loss_minus = self._interpolate_loss(targets, pos_minus)

        # Finite difference gradient
        gradient = (loss_plus - loss_minus) / (2 * epsilon)
        return gradient

    def _interpolate_loss(self, targets: List[Target], position: float) -> float:
        """
        Interpolate loss at a continuous position.

        Linear interpolation between discrete targets.
        """
        # Clamp position
        position = np.clip(position, 0, len(targets) - 1)

        # Get surrounding indices
        idx_low = int(np.floor(position))
        idx_high = int(np.ceil(position))

        if idx_low == idx_high:
            # Exact position
            return self.compute_loss(targets[idx_low].metrics)

        # Interpolate
        weight = position - idx_low
        loss_low = self.compute_loss(targets[idx_low].metrics)
        loss_high = self.compute_loss(targets[idx_high].metrics)

        return (1 - weight) * loss_low + weight * loss_high

    def step(
        self,
        targets: List[Target],
        position: float
    ) -> Tuple[float, float]:
        """
        Single gradient descent step.

        Implements: dθ/dt = -∇L(θ) + η·ξ(t)

        Args:
            targets: List of routing targets
            position: Current position

        Returns:
            (new_position, gradient)
        """
        # Compute gradient
        gradient = self.compute_gradient(targets, position)

        # Gradient descent: move opposite to gradient (downhill)
        descent_term = -self.learning_rate * gradient

        # Add Gaussian noise for exploration
        noise_term = self.noise_level * random.gauss(0, 1)

        # Update position
        delta = (descent_term + noise_term) * self.dt
        new_position = position + delta

        # Clamp to valid range
        new_position = np.clip(new_position, 0, len(targets) - 1)

        return new_position, gradient

    def route(
        self,
        targets: List[str],
        target_metrics: List[Dict[str, float]],
        max_steps: int = 10,
        initial_position: Optional[float] = None
    ) -> RouteDecision:
        """
        Route query by following gradient downhill.

        Args:
            targets: List of target names
            target_metrics: List of metric dicts for each target
            max_steps: Maximum descent iterations
            initial_position: Starting position (defaults to middle)

        Returns:
            RouteDecision with selected target
        """
        # Create Target objects
        target_objs = [
            Target(name=name, metrics=metrics)
            for name, metrics in zip(targets, target_metrics)
        ]

        # Initialize position
        if initial_position is None:
            position = len(targets) / 2.0  # Start in middle
        else:
            position = initial_position

        # Flow downhill for max_steps
        for step in range(max_steps):
            position, gradient = self.step(target_objs, position)

        # Round to nearest target
        target_idx = int(round(position))
        target_idx = np.clip(target_idx, 0, len(targets) - 1)

        # Compute final loss
        final_loss = self.compute_loss(target_objs[target_idx].metrics)

        return RouteDecision(
            target=targets[target_idx],
            loss=final_loss,
            gradient=gradient,
            position=position,
            iterations=max_steps
        )


# Common loss functions

def load_loss(metrics: Dict[str, float]) -> float:
    """Loss based on server load only."""
    return metrics.get("load", 0.0)


def latency_loss(metrics: Dict[str, float]) -> float:
    """Loss based on latency only."""
    latency = metrics.get("latency", 0.0)
    return latency / 100.0  # Normalize to 0-1


def cost_loss(metrics: Dict[str, float]) -> float:
    """Loss based on cost only."""
    return metrics.get("cost", 0.0)


def combined_loss(
    load_weight: float = 0.5,
    latency_weight: float = 0.3,
    cost_weight: float = 0.2
) -> Callable[[Dict[str, float]], float]:
    """
    Create combined loss function.

    Args:
        load_weight: Weight for load (0.0 - 1.0)
        latency_weight: Weight for latency
        cost_weight: Weight for cost

    Returns:
        Loss function combining all metrics
    """
    def loss_fn(metrics: Dict[str, float]) -> float:
        load = metrics.get("load", 0.0)
        latency = metrics.get("latency", 0.0) / 100.0  # Normalize
        cost = metrics.get("cost", 0.0)

        total = (
            load_weight * load +
            latency_weight * latency +
            cost_weight * cost
        )
        return total

    return loss_fn


def quality_loss(metrics: Dict[str, float]) -> float:
    """
    Loss based on quality (inverted - lower quality = higher loss).

    Useful for tool selection where quality matters.
    """
    quality = metrics.get("quality", 1.0)
    return 1.0 - quality  # Invert: high quality = low loss


def create_tool_selection_loss(
    cost_weight: float = 0.3,
    quality_weight: float = 0.5,
    latency_weight: float = 0.2
) -> Callable[[Dict[str, float]], float]:
    """
    Loss function for tool selection.

    Balances cost, quality, and speed.

    Args:
        cost_weight: Weight for tool cost
        quality_weight: Weight for result quality (inverted)
        latency_weight: Weight for execution time

    Returns:
        Loss function for tool selection
    """
    def loss_fn(metrics: Dict[str, float]) -> float:
        cost = metrics.get("cost", 0.0)
        quality = metrics.get("quality", 1.0)
        latency = metrics.get("latency", 0.0) / 200.0  # Normalize to 0-1

        # Invert quality: high quality = low loss
        quality_loss_val = 1.0 - quality

        total = (
            cost_weight * cost +
            quality_weight * quality_loss_val +
            latency_weight * latency
        )
        return total

    return loss_fn
