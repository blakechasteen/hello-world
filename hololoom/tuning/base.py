"""
Base classes for self-tuning system.

Provides:
- ThompsonBandit: Core Thompson Sampling implementation
- TuningAgent: Abstract base for tuning agents
- SafeParameter: Parameter with bounded ranges
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ThompsonBandit:
    """
    Thompson Sampling multi-armed bandit.

    Uses Beta distributions to model arm rewards:
    - α (alpha): Successes + 1
    - β (beta): Failures + 1
    - Sample from Beta(α, β) for each arm, select highest

    Example:
        bandit = ThompsonBandit(n_arms=5)
        arm = bandit.sample()
        # ... execute with arm ...
        bandit.update(arm, success=True, confidence=0.85)
    """

    n_arms: int
    alpha: np.ndarray = field(default_factory=lambda: None)
    beta: np.ndarray = field(default_factory=lambda: None)
    pulls: np.ndarray = field(default_factory=lambda: None)
    total_reward: np.ndarray = field(default_factory=lambda: None)

    def __post_init__(self):
        """Initialize Beta distribution parameters."""
        if self.alpha is None:
            self.alpha = np.ones(self.n_arms)
        if self.beta is None:
            self.beta = np.ones(self.n_arms)
        if self.pulls is None:
            self.pulls = np.zeros(self.n_arms)
        if self.total_reward is None:
            self.total_reward = np.zeros(self.n_arms)

    def sample(self) -> int:
        """Sample arm using Thompson Sampling."""
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm_idx: int, success: bool, confidence: float = 1.0):
        """
        Update arm statistics.

        Args:
            arm_idx: Arm index to update
            success: Whether action succeeded
            confidence: Reward value (0-1), defaults to 1.0
        """
        self.pulls[arm_idx] += 1

        if success:
            self.alpha[arm_idx] += confidence
            self.total_reward[arm_idx] += confidence
        else:
            self.beta[arm_idx] += (1.0 - confidence)

    def get_expected_rewards(self) -> np.ndarray:
        """Get expected reward for each arm (mean of Beta distribution)."""
        return self.alpha / (self.alpha + self.beta)

    def get_stats(self) -> dict[int, dict[str, float]]:
        """Get statistics for all arms."""
        expected = self.get_expected_rewards()

        stats = {}
        for i in range(self.n_arms):
            stats[i] = {
                'alpha': float(self.alpha[i]),
                'beta': float(self.beta[i]),
                'pulls': int(self.pulls[i]),
                'total_reward': float(self.total_reward[i]),
                'expected_reward': float(expected[i]),
                'avg_reward': float(self.total_reward[i] / self.pulls[i]) if self.pulls[i] > 0 else 0.0,
            }

        return stats

    def get_state(self) -> dict[str, Any]:
        """Serialize bandit state for persistence."""
        return {
            'n_arms': self.n_arms,
            'alpha': self.alpha.tolist(),
            'beta': self.beta.tolist(),
            'pulls': self.pulls.tolist(),
            'total_reward': self.total_reward.tolist(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> 'ThompsonBandit':
        """Restore bandit from serialized state."""
        return cls(
            n_arms=state['n_arms'],
            alpha=np.array(state['alpha']),
            beta=np.array(state['beta']),
            pulls=np.array(state['pulls']),
            total_reward=np.array(state['total_reward']),
        )


@dataclass
class SafeParameter:
    """
    Parameter with safety constraints.

    Ensures tuned values stay within safe bounds and change gradually.
    """

    name: str
    current_value: float
    min_value: float
    max_value: float
    max_change_percent: float = 0.2  # Max 20% change per cycle

    def propose(self, new_value: float) -> float:
        """
        Propose new value with safety checks.

        Args:
            new_value: Proposed new value

        Returns:
            Safe value (clipped to bounds and max change)
        """
        # Clip to absolute bounds
        safe_value = np.clip(new_value, self.min_value, self.max_value)

        # Enforce gradual change
        max_change = self.current_value * self.max_change_percent
        max_allowed = self.current_value + max_change
        min_allowed = self.current_value - max_change

        safe_value = np.clip(safe_value, min_allowed, max_allowed)

        return safe_value

    def update(self, new_value: float) -> float:
        """Update parameter value with safety checks."""
        safe_value = self.propose(new_value)
        old_value = self.current_value
        self.current_value = safe_value

        if abs(safe_value - old_value) > 0.001:
            logger.info(f"Parameter {self.name}: {old_value:.3f} → {safe_value:.3f}")

        return safe_value


class TuningAgent(ABC):
    """
    Abstract base class for tuning agents.

    Each agent:
    - Measures performance metrics
    - Explores parameter space using Thompson Sampling
    - Learns from outcomes
    - Persists learned parameters

    Subclasses must implement:
    - measure_performance(): Collect metrics
    - tune_parameters(): Adjust parameters
    - get_state() / load_state(): Persistence
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"tuning.{name}")
        self.total_tuning_cycles = 0
        self.successful_tunings = 0
        self.metrics_history = deque(maxlen=1000)

    @abstractmethod
    async def measure_performance(self) -> dict[str, float]:
        """
        Measure current system performance.

        Returns:
            Dict of metric_name → value
        """
        pass

    @abstractmethod
    async def tune_parameters(self) -> dict[str, Any]:
        """
        Tune parameters based on current performance.

        Returns:
            Dict describing tuning actions taken
        """
        pass

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Serialize agent state for persistence."""
        pass

    @abstractmethod
    def load_state(self, state: dict[str, Any]):
        """Restore agent from serialized state."""
        pass

    async def run_tuning_cycle(self) -> dict[str, Any]:
        """
        Run one complete tuning cycle.

        Returns:
            Tuning result with metrics and actions
        """
        self.total_tuning_cycles += 1

        # Measure baseline
        baseline = await self.measure_performance()

        # Tune parameters
        tuning_result = await self.tune_parameters()

        # Measure after tuning
        new_metrics = await self.measure_performance()

        # Calculate impact
        impact = self._calculate_impact(baseline, new_metrics)

        # Track success
        if impact > 0:
            self.successful_tunings += 1

        # Store metrics
        self.metrics_history.append({
            'timestamp': time.time(),
            'baseline': baseline,
            'new_metrics': new_metrics,
            'impact': impact,
            'tuning_result': tuning_result,
        })

        result = {
            'agent': self.name,
            'cycle': self.total_tuning_cycles,
            'baseline': baseline,
            'new_metrics': new_metrics,
            'impact': impact,
            'success': impact > 0,
            'actions': tuning_result,
        }

        self.logger.info(f"Tuning cycle {self.total_tuning_cycles}: impact={impact:.3f}")

        return result

    def _calculate_impact(self, baseline: dict[str, float], new: dict[str, float]) -> float:
        """
        Calculate tuning impact score.

        Override in subclass for custom impact calculation.
        Default: average relative improvement across all metrics.
        """
        improvements = []

        for metric, baseline_value in baseline.items():
            if metric in new:
                new_value = new[metric]

                # Handle different metric types
                if 'latency' in metric or 'duration' in metric:
                    # Lower is better
                    improvement = (baseline_value - new_value) / baseline_value
                else:
                    # Higher is better
                    improvement = (new_value - baseline_value) / baseline_value if baseline_value > 0 else 0

                improvements.append(improvement)

        return np.mean(improvements) if improvements else 0.0

    def get_success_rate(self) -> float:
        """Get tuning success rate."""
        if self.total_tuning_cycles == 0:
            return 0.0
        return self.successful_tunings / self.total_tuning_cycles


# Safe parameter ranges for common parameters
SAFE_RANGES = {
    # Timeouts (seconds)
    'pipeline_timeout': (0.5, 30.0),
    'retrieval_timeout': (0.05, 10.0),
    'stage_timeout': (0.05, 10.0),

    # Cache sizes
    'cache_size': (100, 1000000),
    'working_memory_size': (10, 10000),

    # Thresholds (0-1)
    'similarity_threshold': (0.0, 1.0),
    'activation_threshold': (0.0, 1.0),
    'confidence_threshold': (0.5, 0.95),

    # Policy parameters
    'epsilon': (0.01, 0.5),
    'learning_rate': (1e-5, 1e-2),
    'blend_weight': (0.0, 1.0),

    # Retrieval
    'retrieval_k': (1, 50),
    'bm25_weight': (0.0, 1.0),

    # Physics
    'spring_stiffness': (0.05, 0.5),
    'spring_damping': (0.5, 0.99),
    'spring_decay': (0.90, 0.99),
}


def create_safe_parameter(name: str, current_value: float,
                         max_change_percent: float = 0.2) -> SafeParameter:
    """
    Create SafeParameter with automatic range lookup.

    Args:
        name: Parameter name
        current_value: Current value
        max_change_percent: Maximum change per cycle (default 20%)

    Returns:
        SafeParameter instance
    """
    if name not in SAFE_RANGES:
        raise ValueError(f"Unknown parameter: {name}. Add to SAFE_RANGES.")

    min_val, max_val = SAFE_RANGES[name]

    return SafeParameter(
        name=name,
        current_value=current_value,
        min_value=min_val,
        max_value=max_val,
        max_change_percent=max_change_percent,
    )
