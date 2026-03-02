"""
HoloLoom Physics - Phase 3: Thermodynamic Optimization

Implements free energy minimization for intelligent exploration/exploitation balance.

Physics Model:
    Helmholtz Free Energy: F = E - T*S

    Where:
    - F: Free energy (objective to minimize)
    - E: Internal energy (cost, error, loss)
    - T: Temperature (exploration parameter)
    - S: Entropy (diversity, uncertainty)

Applications:
    1. Exploration vs Exploitation: Temperature controls balance
    2. System Health: Free energy tracks overall efficiency
    3. Cost Optimization: Minimize energy while maintaining diversity

Architecture:
    ThermodynamicOptimizer
      +- EnergyCalculator (cost functions)
      +- EntropyCalculator (diversity metrics)
      +- TemperatureScheduler (annealing)
      +- FreeEnergyMinimizer (optimizer)

Usage:
    from hololoom.physics.thermodynamics import ThermodynamicOptimizer

    # Create optimizer
    thermo = ThermodynamicOptimizer(
        initial_temperature=1.0,
        cooling_schedule='exponential',
        cooling_rate=0.95
    )

    # Select action with exploration/exploitation balance
    action = thermo.select_action(
        energy_costs=[0.1, 0.3, 0.5],
        diversity_scores=[0.8, 0.5, 0.2],
        temperature=1.0  # High temp -> exploration
    )

Author: Claude Code (with HoloLoom architecture by Blake)
Date: November 9, 2025
Phase: 3 - Thermodynamic Optimization
"""

from __future__ import annotations

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# Temperature Scheduling
# ============================================================================

class CoolingSchedule(Enum):
    """Annealing schedule types."""
    EXPONENTIAL = "exponential"  # T(t) = T0 * rate^t
    LINEAR = "linear"           # T(t) = T0 - rate*t
    INVERSE = "inverse"          # T(t) = T0 / (1 + rate*t)
    ADAPTIVE = "adaptive"        # Adjusts based on system performance


@dataclass
class TemperatureState:
    """Current temperature state."""
    current_temp: float           # Current temperature
    initial_temp: float           # Initial temperature (T0)
    timestep: int                 # Current timestep
    schedule: CoolingSchedule     # Cooling schedule type
    rate: float                   # Cooling rate parameter
    min_temp: float = 0.01        # Minimum temperature (prevents full exploitation)
    max_temp: float = 10.0        # Maximum temperature (prevents infinite exploration)


class TemperatureScheduler:
    """
    Temperature scheduler for simulated annealing.

    Implements multiple cooling schedules:
    - Exponential: Fast cooling (T = T0 * rate^t)
    - Linear: Slow cooling (T = T0 - rate*t)
    - Inverse: Gradual cooling (T = T0 / (1 + rate*t))
    - Adaptive: Adjusts based on performance
    """

    def __init__(
        self,
        initial_temperature: float = 1.0,
        cooling_schedule: str = "exponential",
        cooling_rate: float = 0.95,
        min_temp: float = 0.01,
        max_temp: float = 10.0
    ):
        """
        Initialize temperature scheduler.

        Args:
            initial_temperature: Starting temperature (exploration level)
            cooling_schedule: Schedule type (exponential/linear/inverse/adaptive)
            cooling_rate: Rate parameter for cooling
            min_temp: Minimum temperature (full exploitation)
            max_temp: Maximum temperature (full exploration)
        """
        self.schedule = CoolingSchedule(cooling_schedule)
        self.state = TemperatureState(
            current_temp=initial_temperature,
            initial_temp=initial_temperature,
            timestep=0,
            schedule=self.schedule,
            rate=cooling_rate,
            min_temp=min_temp,
            max_temp=max_temp
        )

    def step(self) -> float:
        """
        Update temperature for next timestep.

        Returns:
            New temperature
        """
        self.state.timestep += 1

        if self.state.schedule == CoolingSchedule.EXPONENTIAL:
            # T(t) = T0 * rate^t
            new_temp = self.state.initial_temp * (self.state.rate ** self.state.timestep)

        elif self.state.schedule == CoolingSchedule.LINEAR:
            # T(t) = T0 - rate*t
            new_temp = self.state.initial_temp - self.state.rate * self.state.timestep

        elif self.state.schedule == CoolingSchedule.INVERSE:
            # T(t) = T0 / (1 + rate*t)
            new_temp = self.state.initial_temp / (1.0 + self.state.rate * self.state.timestep)

        else:  # ADAPTIVE
            # Placeholder for adaptive schedule (would adjust based on performance)
            new_temp = self.state.current_temp * self.state.rate

        # Clamp to [min_temp, max_temp]
        new_temp = max(self.state.min_temp, min(new_temp, self.state.max_temp))

        self.state.current_temp = new_temp
        return new_temp

    def get_temperature(self) -> float:
        """Get current temperature."""
        return self.state.current_temp

    def set_temperature(self, temp: float):
        """Manually set temperature (for adaptive control)."""
        self.state.current_temp = max(
            self.state.min_temp,
            min(temp, self.state.max_temp)
        )

    def reset(self):
        """Reset to initial temperature."""
        self.state.current_temp = self.state.initial_temp
        self.state.timestep = 0


# ============================================================================
# Energy and Entropy Calculation
# ============================================================================

class EnergyCalculator:
    """
    Calculates internal energy (cost, error, loss).

    Energy represents the "cost" of a state:
    - High energy = expensive, error-prone, inefficient
    - Low energy = cheap, accurate, efficient
    """

    @staticmethod
    def compute_energy(
        cost: float = 0.0,
        error: float = 0.0,
        latency: float = 0.0,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute internal energy from multiple cost components.

        Args:
            cost: Monetary cost (0.0-1.0)
            error: Error rate (0.0-1.0)
            latency: Latency penalty (0.0-1.0)
            weights: Component weights (default: equal)

        Returns:
            Total energy E
        """
        if weights is None:
            weights = {"cost": 0.33, "error": 0.33, "latency": 0.34}

        energy = (
            weights.get("cost", 0.33) * cost +
            weights.get("error", 0.33) * error +
            weights.get("latency", 0.34) * latency
        )

        return energy

    @staticmethod
    def compute_action_energies(
        actions: List[str],
        costs: Dict[str, float],
        errors: Dict[str, float],
        latencies: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Compute energy for each action.

        Args:
            actions: List of action names
            costs: Cost per action
            errors: Error rate per action
            latencies: Latency per action
            weights: Component weights

        Returns:
            Energy per action
        """
        energies = {}
        for action in actions:
            energies[action] = EnergyCalculator.compute_energy(
                cost=costs.get(action, 0.0),
                error=errors.get(action, 0.0),
                latency=latencies.get(action, 0.0),
                weights=weights
            )
        return energies


class EntropyCalculator:
    """
    Calculates entropy (diversity, uncertainty).

    Entropy represents the "disorder" or "exploration" of a state:
    - High entropy = diverse, exploratory, uncertain
    - Low entropy = focused, exploitative, certain
    """

    @staticmethod
    def shannon_entropy(probabilities: np.ndarray) -> float:
        """
        Shannon entropy: H = -sum(p * log(p))

        Args:
            probabilities: Probability distribution (must sum to 1)

        Returns:
            Entropy S
        """
        # Filter out zero probabilities (log(0) undefined)
        probs = probabilities[probabilities > 0]

        if len(probs) == 0:
            return 0.0

        # H = -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs))
        return entropy

    @staticmethod
    def diversity_score(
        actions: List[str],
        action_counts: Dict[str, int]
    ) -> float:
        """
        Diversity score based on action distribution.

        High diversity = many actions used equally
        Low diversity = few actions dominate

        Args:
            actions: List of all possible actions
            action_counts: How many times each action was used

        Returns:
            Diversity score (0.0-1.0)
        """
        total = sum(action_counts.values())
        if total == 0:
            return 1.0  # Maximum diversity (uniform)

        # Compute probability distribution
        probs = np.array([
            action_counts.get(action, 0) / total
            for action in actions
        ])

        # Normalize Shannon entropy to [0, 1]
        # Max entropy for uniform distribution: log(N)
        max_entropy = math.log(len(actions))
        normalized_entropy = EntropyCalculator.shannon_entropy(probs) / max_entropy

        return normalized_entropy


# ============================================================================
# Free Energy Minimization
# ============================================================================

@dataclass
class ThermodynamicState:
    """State of thermodynamic system."""
    energy: float                 # Internal energy E
    entropy: float                # Entropy S
    temperature: float            # Temperature T
    free_energy: float           # Free energy F = E - T*S
    timestep: int                # Current timestep
    action_counts: Dict[str, int] = field(default_factory=dict)  # Action history


class ThermodynamicOptimizer:
    """
    Thermodynamic optimizer for exploration/exploitation balance.

    Uses Helmholtz free energy minimization: F = E - T*S

    High temperature (T → ∞):
        F ≈ -T*S (entropy dominates)
        → Explore (maximize diversity)

    Low temperature (T → 0):
        F ≈ E (energy dominates)
        → Exploit (minimize cost)

    Usage:
        thermo = ThermodynamicOptimizer(initial_temperature=1.0)
        action = thermo.select_action(energies, temperature=1.0)
        thermo.update(action, reward=0.8)
    """

    def __init__(
        self,
        initial_temperature: float = 1.0,
        cooling_schedule: str = "exponential",
        cooling_rate: float = 0.95,
        auto_anneal: bool = True
    ):
        """
        Initialize thermodynamic optimizer.

        Args:
            initial_temperature: Starting temperature (exploration level)
            cooling_schedule: Annealing schedule (exponential/linear/inverse/adaptive)
            cooling_rate: Cooling rate parameter
            auto_anneal: Automatically cool temperature over time
        """
        self.scheduler = TemperatureScheduler(
            initial_temperature=initial_temperature,
            cooling_schedule=cooling_schedule,
            cooling_rate=cooling_rate
        )
        self.auto_anneal = auto_anneal

        self.state = ThermodynamicState(
            energy=0.0,
            entropy=1.0,  # High initial entropy (exploration)
            temperature=initial_temperature,
            free_energy=0.0,
            timestep=0,
            action_counts={}
        )

    def compute_free_energy(
        self,
        energy: float,
        entropy: float,
        temperature: float
    ) -> float:
        """
        Compute Helmholtz free energy: F = E - T*S

        Args:
            energy: Internal energy E
            entropy: Entropy S
            temperature: Temperature T

        Returns:
            Free energy F
        """
        return energy - temperature * entropy

    def select_action(
        self,
        actions: List[str],
        energies: Dict[str, float],
        temperature: Optional[float] = None
    ) -> str:
        """
        Select action via Boltzmann distribution.

        Probability: P(a) ∝ exp(-E(a) / T)

        High temp: Uniform (exploration)
        Low temp: Greedy (exploitation)

        Args:
            actions: List of possible actions
            energies: Energy per action
            temperature: Temperature (uses current if None)

        Returns:
            Selected action
        """
        if temperature is None:
            temperature = self.state.temperature

        # Boltzmann distribution: P(a) ∝ exp(-E(a) / T)
        energy_array = np.array([energies[action] for action in actions])
        exp_terms = np.exp(-energy_array / temperature)

        # Normalize to probabilities
        probabilities = exp_terms / exp_terms.sum()

        # Sample action
        action = np.random.choice(actions, p=probabilities)

        return action

    def update(
        self,
        action: str,
        energy: float,
        actions: Optional[List[str]] = None
    ):
        """
        Update thermodynamic state after action.

        Args:
            action: Action taken
            energy: Energy of action
            actions: All possible actions (for entropy calculation)
        """
        # Update action counts
        self.state.action_counts[action] = self.state.action_counts.get(action, 0) + 1

        # Update energy (moving average)
        alpha = 0.1  # Learning rate
        self.state.energy = (1 - alpha) * self.state.energy + alpha * energy

        # Update entropy (diversity of action distribution)
        if actions:
            self.state.entropy = EntropyCalculator.diversity_score(
                actions,
                self.state.action_counts
            )

        # Update temperature (if auto-annealing)
        if self.auto_anneal:
            self.state.temperature = self.scheduler.step()
        else:
            self.state.temperature = self.scheduler.get_temperature()

        # Update free energy
        self.state.free_energy = self.compute_free_energy(
            self.state.energy,
            self.state.entropy,
            self.state.temperature
        )

        self.state.timestep += 1

    def get_statistics(self) -> Dict:
        """Get current thermodynamic statistics."""
        return {
            "energy": self.state.energy,
            "entropy": self.state.entropy,
            "temperature": self.state.temperature,
            "free_energy": self.state.free_energy,
            "timestep": self.state.timestep,
            "action_counts": dict(self.state.action_counts),
            "total_actions": sum(self.state.action_counts.values())
        }


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "CoolingSchedule",
    "TemperatureState",
    "TemperatureScheduler",
    "EnergyCalculator",
    "EntropyCalculator",
    "ThermodynamicState",
    "ThermodynamicOptimizer",
]
