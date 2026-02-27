"""
HoloLoom Physics - Phase 5: Statistical Mechanics

Implements emergent behavior from microscopic interactions using statistical mechanics.

Physics Model:
    Partition Function: Z = Σ exp(-E_i / kT)
    Ensemble Averages: <A> = (1/Z) Σ A_i exp(-E_i / kT)
    Entropy: S = k ln(Ω)  (Boltzmann's formula)
    Free Energy: F = E - T*S

Applications:
    1. Memory Consolidation: Memories cluster via Gibbs distribution
    2. Pattern Emergence: Phase transitions crystallize patterns
    3. Information Temperature: High T = exploration, Low T = exploitation

Architecture:
    StatisticalMechanicsEngine
      +- CanonicalEnsemble (Gibbs distribution)
      +- PhaseTransitionDetector (critical points)
      +- EntropyCalculator (Boltzmann entropy)
      +- FreeEnergyMinimizer (F = E - T*S)

Usage:
    from hololoom.physics.statistical_mechanics import StatisticalMechanicsEngine

    # Create statistical mechanics engine
    engine = StatisticalMechanicsEngine(temperature=1.0)

    # Consolidate memories via Gibbs distribution
    clusters = engine.consolidate_memories(shards)

    # Detect phase transitions (pattern crystallization)
    transition = engine.detect_phase_transition(history)

Author: Claude Code (with HoloLoom architecture by Blake)
Date: November 9, 2025
Phase: 5 - Statistical Mechanics
"""

from __future__ import annotations

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


# ============================================================================
# Statistical Mechanics Constants
# ============================================================================

BOLTZMANN_CONSTANT = 1.0  # k_B in natural units
AVOGADRO_NUMBER = 1.0     # N_A in natural units (for ensemble averaging)


# ============================================================================
# Microstate and Macrostate
# ============================================================================

@dataclass
class Microstate:
    """Individual microscopic state (e.g., single memory configuration)."""
    id: str                          # Unique identifier
    state_vector: np.ndarray         # State representation (embedding)
    energy: float                    # Energy of this state
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Macrostate:
    """Macroscopic state (emergent from microstates)."""
    label: str                       # Cluster/pattern label
    microstates: List[Microstate]    # Constituent microstates
    average_energy: float            # <E> = Σ p_i E_i
    entropy: float                   # S = -k Σ p_i ln(p_i)
    free_energy: float               # F = <E> - T*S
    order_parameter: float           # Measure of order (0=disordered, 1=ordered)


# ============================================================================
# Canonical Ensemble (Gibbs Distribution)
# ============================================================================

class CanonicalEnsemble:
    """
    Canonical ensemble for fixed N, V, T.

    Implements Gibbs distribution: P(i) = exp(-E_i/kT) / Z
    where Z = Σ exp(-E_i/kT) is the partition function.
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize canonical ensemble.

        Args:
            temperature: Thermodynamic temperature (kT in energy units)
        """
        self.temperature = temperature
        self.kT = BOLTZMANN_CONSTANT * temperature

    def partition_function(self, energies: List[float]) -> float:
        """
        Compute partition function Z = Σ exp(-E_i/kT).

        Args:
            energies: List of state energies

        Returns:
            Partition function Z
        """
        if self.kT == 0:
            # T=0 limit: only ground state
            return 1.0

        # Z = Σ exp(-E_i/kT)
        Z = sum(np.exp(-E / self.kT) for E in energies)
        return Z

    def gibbs_probabilities(self, energies: List[float]) -> np.ndarray:
        """
        Compute Gibbs probability distribution P(i) = exp(-E_i/kT) / Z.

        Args:
            energies: List of state energies

        Returns:
            Array of probabilities (sums to 1)
        """
        Z = self.partition_function(energies)

        if Z == 0:
            # Uniform if partition function is zero (shouldn't happen)
            return np.ones(len(energies)) / len(energies)

        # P(i) = exp(-E_i/kT) / Z
        probabilities = np.array([
            np.exp(-E / self.kT) / Z
            for E in energies
        ])

        return probabilities

    def ensemble_average(self, values: List[float], energies: List[float]) -> float:
        """
        Compute ensemble average <A> = Σ A_i P(i).

        Args:
            values: Observable values A_i
            energies: State energies E_i

        Returns:
            Ensemble average <A>
        """
        probabilities = self.gibbs_probabilities(energies)

        # <A> = Σ A_i * P(i)
        average = sum(
            value * prob
            for value, prob in zip(values, probabilities)
        )

        return average

    def free_energy(self, energies: List[float]) -> float:
        """
        Compute Helmholtz free energy F = -kT ln(Z).

        Args:
            energies: List of state energies

        Returns:
            Free energy F
        """
        Z = self.partition_function(energies)

        if Z <= 0:
            return float('inf')  # Infinite free energy if Z=0

        # F = -kT ln(Z)
        F = -self.kT * np.log(Z)

        return F


# ============================================================================
# Entropy Calculation
# ============================================================================

class EntropyCalculator:
    """
    Calculate entropy via multiple formulations.

    1. Gibbs Entropy: S = -k Σ p_i ln(p_i)  (information-theoretic)
    2. Boltzmann Entropy: S = k ln(Ω)  (combinatorial)
    """

    @staticmethod
    def gibbs_entropy(probabilities: np.ndarray) -> float:
        """
        Compute Gibbs entropy S = -k Σ p_i ln(p_i).

        Args:
            probabilities: Probability distribution

        Returns:
            Gibbs entropy
        """
        # Filter out zero probabilities (ln(0) undefined)
        nonzero = probabilities[probabilities > 0]

        if len(nonzero) == 0:
            return 0.0

        # S = -k Σ p_i ln(p_i)
        S = -BOLTZMANN_CONSTANT * sum(
            p * np.log(p)
            for p in nonzero
        )

        return S

    @staticmethod
    def boltzmann_entropy(num_microstates: int) -> float:
        """
        Compute Boltzmann entropy S = k ln(Ω).

        Args:
            num_microstates: Number of accessible microstates Ω

        Returns:
            Boltzmann entropy
        """
        if num_microstates <= 0:
            return 0.0

        # S = k ln(Ω)
        S = BOLTZMANN_CONSTANT * np.log(num_microstates)

        return S

    @staticmethod
    def mixing_entropy(fractions: List[float]) -> float:
        """
        Compute mixing entropy S_mix = -k Σ x_i ln(x_i).

        Args:
            fractions: Mole fractions x_i

        Returns:
            Mixing entropy
        """
        # Same as Gibbs entropy
        return EntropyCalculator.gibbs_entropy(np.array(fractions))


# ============================================================================
# Phase Transition Detection
# ============================================================================

class PhaseType(Enum):
    """Type of phase transition."""
    FIRST_ORDER = "first_order"      # Discontinuous (e.g., water → ice)
    SECOND_ORDER = "second_order"    # Continuous (e.g., ferromagnet)
    CROSSOVER = "crossover"          # Smooth change


@dataclass
class PhaseTransition:
    """Detected phase transition."""
    critical_temperature: float      # T_c where transition occurs
    phase_type: PhaseType           # Type of transition
    order_before: float             # Order parameter before transition
    order_after: float              # Order parameter after transition
    latent_heat: float = 0.0       # Energy released (first-order only)
    critical_exponent: float = 0.0  # Power law exponent (second-order)


class PhaseTransitionDetector:
    """
    Detect phase transitions via order parameter analysis.

    Phase transitions occur when system behavior changes qualitatively:
    - First-order: Discontinuous jump (e.g., clustering → single cluster)
    - Second-order: Continuous but non-analytic (e.g., power law scaling)
    """

    def __init__(self, history_window: int = 50):
        """
        Initialize phase transition detector.

        Args:
            history_window: Number of states to analyze
        """
        self.history_window = history_window

    def compute_order_parameter(self, microstates: List[Microstate]) -> float:
        """
        Compute order parameter (measure of system organization).

        For clustering: variance of inter-cluster distances
        Low variance = ordered (clustered), High variance = disordered

        Args:
            microstates: List of microstates

        Returns:
            Order parameter (0=disordered, 1=ordered)
        """
        if len(microstates) < 2:
            return 0.0

        # Extract state vectors
        vectors = np.array([state.state_vector for state in microstates])

        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(vectors, metric='euclidean')

        # Order parameter: 1 - (std / mean)
        # High std = disordered, Low std = ordered
        if distances.mean() == 0:
            return 1.0

        order = max(0.0, 1.0 - distances.std() / distances.mean())

        return order

    def detect_transition(
        self,
        history: List[List[Microstate]],
        temperature_history: List[float]
    ) -> Optional[PhaseTransition]:
        """
        Detect phase transition in state history.

        Args:
            history: List of state configurations over time
            temperature_history: Corresponding temperatures

        Returns:
            PhaseTransition if detected, None otherwise
        """
        if len(history) < self.history_window:
            return None  # Not enough data

        # Compute order parameter for each state
        order_params = [
            self.compute_order_parameter(states)
            for states in history
        ]

        # Detect discontinuity (first-order transition)
        for i in range(1, len(order_params)):
            delta_order = abs(order_params[i] - order_params[i-1])

            # Large jump = first-order transition
            if delta_order > 0.3:  # Threshold for discontinuity
                return PhaseTransition(
                    critical_temperature=temperature_history[i],
                    phase_type=PhaseType.FIRST_ORDER,
                    order_before=order_params[i-1],
                    order_after=order_params[i],
                    latent_heat=delta_order  # Proportional to jump
                )

        # Detect critical point (second-order transition)
        # Look for non-analytic behavior (peak in susceptibility)
        derivatives = np.diff(order_params)
        second_derivatives = np.diff(derivatives)

        if len(second_derivatives) > 0:
            peak_idx = np.argmax(np.abs(second_derivatives))

            # Significant curvature change = critical point
            if abs(second_derivatives[peak_idx]) > 0.1:
                return PhaseTransition(
                    critical_temperature=temperature_history[peak_idx + 2],
                    phase_type=PhaseType.SECOND_ORDER,
                    order_before=order_params[peak_idx],
                    order_after=order_params[peak_idx + 2],
                    critical_exponent=self._estimate_critical_exponent(
                        order_params[peak_idx:peak_idx+10],
                        temperature_history[peak_idx:peak_idx+10]
                    )
                )

        return None

    def _estimate_critical_exponent(
        self,
        order_params: List[float],
        temperatures: List[float]
    ) -> float:
        """
        Estimate critical exponent β from power law scaling.

        Near critical point: m ~ |T - T_c|^β

        Args:
            order_params: Order parameter values
            temperatures: Temperature values

        Returns:
            Critical exponent β
        """
        if len(order_params) < 3:
            return 0.5  # Mean field value

        # Fit power law (log-log linear fit)
        # log(m) = β log(|T - T_c|) + const

        T_c = temperatures[0]  # Assume first point is critical
        delta_T = np.abs(np.array(temperatures) - T_c) + 1e-10  # Avoid log(0)

        # Linear regression in log-log space
        log_dT = np.log(delta_T)
        log_m = np.log(np.abs(order_params) + 1e-10)

        # β = slope of log(m) vs log(|T-Tc|)
        beta = np.polyfit(log_dT, log_m, 1)[0]

        return abs(beta)


# ============================================================================
# Memory Consolidation via Statistical Mechanics
# ============================================================================

class StatisticalMechanicsEngine:
    """
    Statistical mechanics engine for emergent memory consolidation.

    Uses Gibbs distribution to naturally cluster memories by energy landscape.
    Detects phase transitions where patterns crystallize.
    """

    def __init__(self, temperature: float = 1.0, cooling_rate: float = 0.95):
        """
        Initialize statistical mechanics engine.

        Args:
            temperature: Initial temperature
            cooling_rate: Temperature decay rate (for annealing)
        """
        self.temperature = temperature
        self.cooling_rate = cooling_rate

        self.ensemble = CanonicalEnsemble(temperature)
        self.entropy_calc = EntropyCalculator()
        self.phase_detector = PhaseTransitionDetector()

        self.state_history: List[List[Microstate]] = []
        self.temperature_history: List[float] = []

    def compute_energy(self, state: Microstate, context: Optional[List[Microstate]] = None) -> float:
        """
        Compute energy of a microstate.

        Energy = -similarity to context (lower energy = better fit)

        Args:
            state: Microstate to evaluate
            context: Context microstates (for interaction energy)

        Returns:
            Energy (lower = more stable)
        """
        if context is None or len(context) == 0:
            # Isolated energy (intrinsic)
            return np.linalg.norm(state.state_vector)

        # Interaction energy: -Σ similarity(state, context_i)
        energy = 0.0
        for other in context:
            # Cosine similarity
            similarity = np.dot(state.state_vector, other.state_vector) / (
                np.linalg.norm(state.state_vector) * np.linalg.norm(other.state_vector) + 1e-10
            )
            energy -= similarity  # Negative similarity = lower energy for similar states

        return energy / len(context)

    def consolidate_memories(
        self,
        microstates: List[Microstate],
        num_clusters: Optional[int] = None
    ) -> List[Macrostate]:
        """
        Consolidate memories via Gibbs distribution clustering.

        Memories with similar energies naturally cluster together.

        Args:
            microstates: Individual memory states
            num_clusters: Target number of clusters (None = auto)

        Returns:
            List of macrostates (clusters)
        """
        if len(microstates) == 0:
            return []

        # Compute energies
        energies = [
            self.compute_energy(state, microstates)
            for state in microstates
        ]

        # Update energy in microstates
        for state, energy in zip(microstates, energies):
            state.energy = energy

        # Gibbs probabilities
        probabilities = self.ensemble.gibbs_probabilities(energies)

        # Entropy of current distribution
        entropy = self.entropy_calc.gibbs_entropy(probabilities)

        # Cluster by energy landscape
        clusters = self._cluster_by_energy(microstates, energies, num_clusters)

        # Create macrostates
        macrostates = []
        for cluster_id, cluster_states in enumerate(clusters):
            cluster_energies = [s.energy for s in cluster_states]
            avg_energy = np.mean(cluster_energies)

            # Cluster entropy (within cluster)
            cluster_probs = self.ensemble.gibbs_probabilities(cluster_energies)
            cluster_entropy = self.entropy_calc.gibbs_entropy(cluster_probs)

            # Free energy of cluster
            F = avg_energy - self.temperature * cluster_entropy

            # Order parameter (how organized is this cluster)
            order = self.phase_detector.compute_order_parameter(cluster_states)

            macrostate = Macrostate(
                label=f"cluster_{cluster_id}",
                microstates=cluster_states,
                average_energy=avg_energy,
                entropy=cluster_entropy,
                free_energy=F,
                order_parameter=order
            )

            macrostates.append(macrostate)

        # Record history for phase transition detection
        self.state_history.append(microstates)
        self.temperature_history.append(self.temperature)

        return macrostates

    def _cluster_by_energy(
        self,
        microstates: List[Microstate],
        energies: List[float],
        num_clusters: Optional[int] = None
    ) -> List[List[Microstate]]:
        """
        Cluster microstates by energy landscape.

        Uses k-means clustering in energy space.

        Args:
            microstates: States to cluster
            energies: State energies
            num_clusters: Number of clusters (None = auto-detect)

        Returns:
            List of clusters (each cluster is list of microstates)
        """
        if num_clusters is None:
            # Auto-detect: sqrt(N)
            num_clusters = max(1, int(np.sqrt(len(microstates))))

        # Simple k-means clustering by energy
        from sklearn.cluster import KMeans

        # Cluster based on state vectors
        vectors = np.array([s.state_vector for s in microstates])
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(vectors)

        # Group by cluster
        clusters = [[] for _ in range(num_clusters)]
        for state, label in zip(microstates, labels):
            clusters[label].append(state)

        # Filter empty clusters
        clusters = [c for c in clusters if len(c) > 0]

        return clusters

    def detect_phase_transition(self) -> Optional[PhaseTransition]:
        """
        Detect phase transition in state history.

        Returns:
            PhaseTransition if detected, None otherwise
        """
        if len(self.state_history) < 10:
            return None  # Not enough history

        return self.phase_detector.detect_transition(
            self.state_history,
            self.temperature_history
        )

    def anneal(self):
        """Cool temperature (simulated annealing)."""
        self.temperature *= self.cooling_rate
        self.ensemble.temperature = self.temperature
        self.ensemble.kT = BOLTZMANN_CONSTANT * self.temperature

    def get_statistics(self) -> Dict:
        """Get statistical mechanics statistics."""
        if len(self.state_history) == 0:
            return {
                "temperature": self.temperature,
                "num_states": 0,
                "entropy": 0.0,
                "free_energy": 0.0
            }

        # Current state
        current_states = self.state_history[-1]
        energies = [s.energy for s in current_states]

        # Probabilities and entropy
        probs = self.ensemble.gibbs_probabilities(energies)
        entropy = self.entropy_calc.gibbs_entropy(probs)

        # Free energy
        F = self.ensemble.free_energy(energies)

        return {
            "temperature": self.temperature,
            "num_states": len(current_states),
            "entropy": entropy,
            "free_energy": F,
            "average_energy": np.mean(energies),
            "history_length": len(self.state_history)
        }


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "Microstate",
    "Macrostate",
    "CanonicalEnsemble",
    "EntropyCalculator",
    "PhaseType",
    "PhaseTransition",
    "PhaseTransitionDetector",
    "StatisticalMechanicsEngine",
    "BOLTZMANN_CONSTANT",
]
