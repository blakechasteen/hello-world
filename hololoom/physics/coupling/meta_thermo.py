"""
Meta-Thermodynamics — the system learns its own temperature.

Thompson Sampling over cooling schedules. The system observes
(T, S, F, order_param) as a state vector and learns which cooling
schedule produces the best outcomes (free energy down, order up).
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .protocol import PhysicsState

logger = logging.getLogger(__name__)


@dataclass
class ScheduleArm:
    """Thompson Sampling arm for a cooling schedule."""
    successes: float = 1.0
    failures: float = 1.0

    def sample(self) -> float:
        return float(np.random.beta(self.successes, self.failures))

    def update(self, reward: float) -> None:
        if reward > 0:
            self.successes += min(reward, 1.0)
        else:
            self.failures += min(abs(reward), 1.0)

    @property
    def mean(self) -> float:
        return self.successes / (self.successes + self.failures)


class MetaThermodynamics:
    """The system learns its own temperature schedule.

    Observes the physics state vector over time and uses Thompson
    Sampling to select the best cooling schedule. The meta-loop:
    system adjusts its own cooling -> outcome changes -> bandit learns.
    """

    SCHEDULES = ("exponential", "linear", "inverse", "adaptive")

    def __init__(self, eval_window: int = 10):
        self._arms: dict[str, ScheduleArm] = {
            s: ScheduleArm() for s in self.SCHEDULES
        }
        self._current: str = "exponential"
        self._history: deque[tuple[float, float, float, float]] = deque(maxlen=200)
        self._eval_window = eval_window
        self._steps_since_eval = 0

    def observe(self, state: PhysicsState) -> float | None:
        """Track state vector. Evaluate every eval_window steps.

        Returns reward if evaluation happened, None otherwise.
        """
        vec = (state.temperature, state.entropy,
               state.free_energy, state.order_parameter)
        self._history.append(vec)
        self._steps_since_eval += 1

        if self._steps_since_eval >= self._eval_window and len(self._history) >= self._eval_window:
            return self._evaluate()
        return None

    def _evaluate(self) -> float:
        """Evaluate: did free energy decrease? Did order increase?"""
        self._steps_since_eval = 0
        recent = list(self._history)[-self._eval_window:]

        F_start, F_end = recent[0][2], recent[-1][2]
        order_start, order_end = recent[0][3], recent[-1][3]

        reward = 0.0
        # Free energy should decrease (system finding minimum)
        if F_end < F_start:
            reward += 0.5
        elif F_end > F_start + 0.1:
            reward -= 0.3

        # Order parameter should increase (structure emerging)
        if order_end > order_start:
            reward += 0.5
        elif order_end < order_start - 0.1:
            reward -= 0.2

        self._arms[self._current].update(reward)
        logger.debug(
            f"Meta-thermo eval: schedule={self._current}, reward={reward:.2f}, "
            f"F: {F_start:.3f}->{F_end:.3f}, order: {order_start:.3f}->{order_end:.3f}"
        )
        return reward

    def select_schedule(self) -> str:
        """Thompson Sample the best cooling schedule."""
        self._current = max(self._arms, key=lambda k: self._arms[k].sample())
        return self._current

    @property
    def current_schedule(self) -> str:
        return self._current

    def apply_schedule(self, state: PhysicsState) -> PhysicsState:
        """Apply the selected cooling schedule to the state temperature."""
        T = state.temperature
        dt = state.dt

        if self._current == "exponential":
            T *= (1 - 0.01 * dt)  # Slow exponential decay
        elif self._current == "linear":
            T = max(0.01, T - 0.005 * dt)
        elif self._current == "inverse":
            T = max(0.01, T / (1 + 0.01 * dt * state.timestep))
        elif self._current == "adaptive":
            # Adaptive: cool faster when order is high, slower when low
            rate = 0.02 * (1 + state.order_parameter)
            T *= (1 - rate * dt)
            T = max(0.01, T)

        state.temperature = T
        return state

    def stats(self) -> dict:
        """Summary for visualization."""
        return {
            "current_schedule": self._current,
            "arms": {
                name: {"mean": round(arm.mean, 3),
                       "successes": round(arm.successes, 1),
                       "failures": round(arm.failures, 1)}
                for name, arm in self._arms.items()
            },
            "history_length": len(self._history),
        }
