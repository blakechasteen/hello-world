"""
Differential Privacy - Privacy-Preserving Computations
=======================================================

Mathematical framework for privacy-preserving data analysis.

Core Concept:
    A mechanism M is (ε,δ)-differentially private if for all S ⊆ Range(M)
    and neighboring datasets D, D' differing in one record:

    P[M(D) ∈ S] ≤ exp(ε) · P[M(D') ∈ S] + δ

Classes:
    PrivacyBudget: Track cumulative privacy loss
    DifferentialPrivacy: Core DP mechanisms
    PrivateAggregator: Privacy-preserving aggregation
    PrivateVectorMean: Private mean of embedding vectors

Mechanisms:
    - Laplace: For ε-DP on numeric queries (L1 sensitivity)
    - Gaussian: For (ε,δ)-DP with better concentration (L2 sensitivity)
    - Exponential: For discrete outputs with utility function

Applications:
    - Private memory retrieval (protect query patterns)
    - Private embedding aggregation (protect individual contributions)
    - Private learning (protect training examples)
    - Private statistics (protect sensitive data)

Author: Claude Code
Date: December 2025 (Math Module Expansion)
"""

import numpy as np
from typing import List, Optional, Callable, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math


class PrivacyMechanism(Enum):
    """Available privacy mechanisms."""
    LAPLACE = "laplace"      # ε-DP, L1 sensitivity
    GAUSSIAN = "gaussian"    # (ε,δ)-DP, L2 sensitivity
    EXPONENTIAL = "exponential"  # Discrete outputs


@dataclass
class PrivacyBudget:
    """
    Track cumulative privacy budget.

    Under composition, privacy degrades:
    - Basic: ε_total = Σ ε_i (linear)
    - Advanced: ε_total = √(2k ln(1/δ)) · ε (for k queries)

    Attributes:
        epsilon: Current privacy parameter (lower = more private)
        delta: Failure probability (0 for pure DP)
        queries_made: Number of queries answered
        history: List of (ε, δ) for each query
    """
    epsilon: float = 0.0
    delta: float = 0.0
    queries_made: int = 0
    history: List[Tuple[float, float]] = field(default_factory=list)

    def add_query(self, epsilon: float, delta: float = 0.0) -> None:
        """Record a privacy expenditure."""
        self.history.append((epsilon, delta))
        self.queries_made += 1

        # Basic composition (can be tightened with advanced composition)
        self.epsilon += epsilon
        self.delta += delta

    @property
    def remaining(self) -> float:
        """Remaining budget (assuming total budget of 1.0)."""
        return max(0.0, 1.0 - self.epsilon)

    def is_exhausted(self, max_epsilon: float = 1.0) -> bool:
        """Check if budget is exhausted."""
        return self.epsilon >= max_epsilon

    def advanced_composition(self, target_delta: float = 1e-5) -> float:
        """
        Compute tighter bound using advanced composition theorem.

        For k (ε,δ)-DP mechanisms:
            Total is (ε', kδ + δ')-DP where
            ε' = √(2k ln(1/δ')) · ε + k·ε·(exp(ε)-1)

        Returns:
            Tighter epsilon bound
        """
        if not self.history:
            return 0.0

        k = len(self.history)
        eps_list = [h[0] for h in self.history]
        avg_eps = np.mean(eps_list)

        # Advanced composition bound
        eps_advanced = (
            np.sqrt(2 * k * np.log(1 / target_delta)) * avg_eps +
            k * avg_eps * (np.exp(avg_eps) - 1)
        )

        return min(eps_advanced, self.epsilon)  # Take tighter bound


def compute_sensitivity(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    norm: str = "L1"
) -> float:
    """
    Estimate sensitivity of function f.

    Sensitivity = max_{x,x' neighbors} ||f(x) - f(x')||

    Args:
        f: Function to analyze
        x: Sample input
        norm: "L1" or "L2"

    Returns:
        Estimated sensitivity
    """
    n = len(x)
    max_sensitivity = 0.0

    # Test removing each element
    for i in range(min(n, 100)):  # Sample for efficiency
        x_prime = np.delete(x, i)

        # Pad to same length for function evaluation
        x_padded = np.zeros_like(x)
        x_padded[:len(x_prime)] = x_prime

        diff = f(x) - f(x_padded)

        if norm == "L1":
            sensitivity = np.sum(np.abs(diff))
        else:  # L2
            sensitivity = np.sqrt(np.sum(diff ** 2))

        max_sensitivity = max(max_sensitivity, sensitivity)

    return max_sensitivity


def laplace_mechanism(
    value: Union[float, np.ndarray],
    sensitivity: float,
    epsilon: float
) -> Union[float, np.ndarray]:
    """
    Laplace mechanism for ε-differential privacy.

    Adds Lap(sensitivity/ε) noise to query output.

    Args:
        value: True query answer
        sensitivity: L1 sensitivity of query
        epsilon: Privacy parameter

    Returns:
        Noisy answer satisfying ε-DP

    Example:
        >>> true_count = 100
        >>> noisy_count = laplace_mechanism(true_count, sensitivity=1, epsilon=0.1)
    """
    scale = sensitivity / epsilon

    if isinstance(value, np.ndarray):
        noise = np.random.laplace(0, scale, value.shape)
    else:
        noise = np.random.laplace(0, scale)

    return value + noise


def gaussian_mechanism(
    value: Union[float, np.ndarray],
    sensitivity: float,
    epsilon: float,
    delta: float = 1e-5
) -> Union[float, np.ndarray]:
    """
    Gaussian mechanism for (ε,δ)-differential privacy.

    Adds N(0, σ²) noise where σ = sensitivity · √(2ln(1.25/δ)) / ε

    Better concentration than Laplace but requires δ > 0.

    Args:
        value: True query answer
        sensitivity: L2 sensitivity of query
        epsilon: Privacy parameter
        delta: Failure probability

    Returns:
        Noisy answer satisfying (ε,δ)-DP
    """
    # Compute noise scale
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    if isinstance(value, np.ndarray):
        noise = np.random.normal(0, sigma, value.shape)
    else:
        noise = np.random.normal(0, sigma)

    return value + noise


def exponential_mechanism(
    candidates: List[Any],
    utility_fn: Callable[[Any], float],
    sensitivity: float,
    epsilon: float
) -> Any:
    """
    Exponential mechanism for discrete outputs.

    Selects output r with probability ∝ exp(ε · u(r) / (2Δu))

    Args:
        candidates: Possible outputs
        utility_fn: u(r) - higher is better
        sensitivity: Δu = max change in utility for neighboring databases
        epsilon: Privacy parameter

    Returns:
        Privately selected output

    Example:
        >>> items = ['A', 'B', 'C']
        >>> utility = {'A': 10, 'B': 8, 'C': 3}
        >>> selected = exponential_mechanism(
        ...     items,
        ...     lambda x: utility[x],
        ...     sensitivity=1,
        ...     epsilon=1.0
        ... )
    """
    # Compute selection probabilities
    utilities = np.array([utility_fn(c) for c in candidates])

    # Scale by privacy parameter
    scaled = epsilon * utilities / (2 * sensitivity)

    # Softmax for numerical stability
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    probs = probs / probs.sum()

    # Sample
    idx = np.random.choice(len(candidates), p=probs)
    return candidates[idx]


def compose_privacy(
    mechanisms: List[Tuple[float, float]],
    composition_type: str = "basic"
) -> Tuple[float, float]:
    """
    Compose multiple privacy guarantees.

    Args:
        mechanisms: List of (ε, δ) tuples
        composition_type: "basic" or "advanced"

    Returns:
        (total_ε, total_δ) for composed mechanism
    """
    if composition_type == "basic":
        # Sequential composition: add up
        total_eps = sum(m[0] for m in mechanisms)
        total_delta = sum(m[1] for m in mechanisms)
        return (total_eps, total_delta)

    else:  # advanced
        # Advanced composition theorem
        k = len(mechanisms)
        eps_list = [m[0] for m in mechanisms]
        delta_list = [m[1] for m in mechanisms]

        avg_eps = np.mean(eps_list)
        target_delta = 1e-5

        total_eps = (
            np.sqrt(2 * k * np.log(1 / target_delta)) * avg_eps +
            k * avg_eps * (np.exp(avg_eps) - 1)
        )
        total_delta = sum(delta_list) + target_delta

        return (total_eps, total_delta)


class DifferentialPrivacy:
    """
    Core differential privacy implementation.

    Provides unified interface for all DP mechanisms with
    automatic budget tracking.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        mechanism: PrivacyMechanism = PrivacyMechanism.GAUSSIAN
    ):
        """
        Args:
            epsilon: Total privacy budget
            delta: Failure probability
            mechanism: Default mechanism to use
        """
        self.total_epsilon = epsilon
        self.total_delta = delta
        self.mechanism = mechanism
        self.budget = PrivacyBudget()

    def query(
        self,
        value: Union[float, np.ndarray],
        sensitivity: float,
        epsilon: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Answer a query with differential privacy.

        Args:
            value: True query answer
            sensitivity: Query sensitivity
            epsilon: Privacy budget for this query (default: 0.1)

        Returns:
            Noisy answer

        Raises:
            ValueError: If privacy budget exhausted
        """
        eps = epsilon or 0.1

        if self.budget.epsilon + eps > self.total_epsilon:
            raise ValueError(
                f"Privacy budget exhausted. "
                f"Used: {self.budget.epsilon:.3f}, "
                f"Requested: {eps:.3f}, "
                f"Total: {self.total_epsilon:.3f}"
            )

        # Apply mechanism
        if self.mechanism == PrivacyMechanism.LAPLACE:
            result = laplace_mechanism(value, sensitivity, eps)
            self.budget.add_query(eps, 0.0)

        elif self.mechanism == PrivacyMechanism.GAUSSIAN:
            delta = self.total_delta / 100  # Divide delta among queries
            result = gaussian_mechanism(value, sensitivity, eps, delta)
            self.budget.add_query(eps, delta)

        else:
            raise ValueError(f"Cannot use {self.mechanism} for numeric queries")

        return result

    def select(
        self,
        candidates: List[Any],
        utility_fn: Callable[[Any], float],
        sensitivity: float,
        epsilon: Optional[float] = None
    ) -> Any:
        """
        Private selection from candidates.

        Uses exponential mechanism.
        """
        eps = epsilon or 0.1

        if self.budget.epsilon + eps > self.total_epsilon:
            raise ValueError("Privacy budget exhausted")

        result = exponential_mechanism(candidates, utility_fn, sensitivity, eps)
        self.budget.add_query(eps, 0.0)

        return result

    @property
    def remaining_budget(self) -> float:
        """Remaining privacy budget."""
        return self.total_epsilon - self.budget.epsilon


class PrivateAggregator:
    """
    Privacy-preserving aggregation for federated scenarios.

    Useful for:
    - Aggregating embeddings from multiple sources
    - Computing statistics without revealing individuals
    - Federated learning with DP guarantees
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0
    ):
        """
        Args:
            epsilon: Privacy budget
            delta: Failure probability
            clip_norm: Maximum L2 norm per contribution
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.dp = DifferentialPrivacy(epsilon, delta, PrivacyMechanism.GAUSSIAN)

    def clip(self, vector: np.ndarray) -> np.ndarray:
        """Clip vector to bounded L2 norm."""
        norm = np.linalg.norm(vector)
        if norm > self.clip_norm:
            return vector * (self.clip_norm / norm)
        return vector

    def aggregate_sum(
        self,
        vectors: List[np.ndarray],
        num_contributors: int
    ) -> np.ndarray:
        """
        Compute private sum of vectors.

        Args:
            vectors: List of contribution vectors
            num_contributors: Number of contributors (for sensitivity)

        Returns:
            Private sum with DP guarantee
        """
        # Clip each contribution
        clipped = [self.clip(v) for v in vectors]

        # True sum
        true_sum = np.sum(clipped, axis=0)

        # Sensitivity is clip_norm (max change from one person)
        sensitivity = self.clip_norm

        # Add noise
        return self.dp.query(true_sum, sensitivity, self.epsilon)

    def aggregate_mean(
        self,
        vectors: List[np.ndarray],
        num_contributors: int
    ) -> np.ndarray:
        """
        Compute private mean of vectors.

        Args:
            vectors: List of contribution vectors
            num_contributors: Number of contributors

        Returns:
            Private mean with DP guarantee
        """
        private_sum = self.aggregate_sum(vectors, num_contributors)
        return private_sum / num_contributors


class PrivateVectorMean:
    """
    Privacy-preserving mean of embedding vectors.

    Integration with HoloLoom:
    - Private aggregation of memory embeddings
    - Privacy-preserving similarity search
    - Federated memory systems
    """

    def __init__(
        self,
        dimension: int,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0
    ):
        """
        Args:
            dimension: Vector dimension
            epsilon: Privacy budget
            delta: Failure probability
            clip_norm: Maximum norm per vector
        """
        self.dimension = dimension
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm

        self.count = 0
        self.sum = np.zeros(dimension)
        self.aggregator = PrivateAggregator(epsilon, delta, clip_norm)

    def add(self, vector: np.ndarray) -> None:
        """Add a vector to the running mean."""
        if len(vector) != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {len(vector)}")

        clipped = self.aggregator.clip(vector)
        self.sum += clipped
        self.count += 1

    def get_private_mean(self) -> np.ndarray:
        """
        Get private mean with DP guarantee.

        Returns:
            Private mean vector
        """
        if self.count == 0:
            return np.zeros(self.dimension)

        true_mean = self.sum / self.count

        # Sensitivity of mean = clip_norm / n
        sensitivity = self.clip_norm / self.count

        # Add Gaussian noise
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(0, sigma, self.dimension)

        return true_mean + noise

    def reset(self) -> None:
        """Reset the accumulator."""
        self.count = 0
        self.sum = np.zeros(self.dimension)


# HoloLoom Integration
class PrivateMemoryRetrieval:
    """
    Privacy-preserving memory retrieval for HoloLoom.

    Protects:
    - Query patterns (what users ask about)
    - Retrieved memories (what knowledge is accessed)
    - Usage statistics (frequency of access)
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5
    ):
        self.dp = DifferentialPrivacy(epsilon, delta, PrivacyMechanism.EXPONENTIAL)
        self.epsilon = epsilon

    def private_top_k(
        self,
        scores: np.ndarray,
        k: int,
        sensitivity: float = 1.0
    ) -> List[int]:
        """
        Private selection of top-k items.

        Uses exponential mechanism to select items with
        probability proportional to their scores.

        Args:
            scores: Relevance scores for each item
            k: Number of items to select
            sensitivity: Score sensitivity

        Returns:
            Indices of selected items (approximately top-k)
        """
        n = len(scores)
        selected = []
        remaining_idx = list(range(n))
        remaining_scores = scores.copy()

        eps_per_selection = self.epsilon / k

        for _ in range(min(k, n)):
            # Select one item privately
            idx = exponential_mechanism(
                remaining_idx,
                lambda i: remaining_scores[remaining_idx.index(i)],
                sensitivity,
                eps_per_selection
            )
            selected.append(idx)

            # Remove from consideration
            pos = remaining_idx.index(idx)
            remaining_idx.pop(pos)
            remaining_scores = np.delete(remaining_scores, pos)

        return selected


if __name__ == "__main__":
    print("Differential Privacy Module Demo")
    print("=" * 50)

    # Demo: Private counting
    print("\n1. Private Counting (Laplace Mechanism)")
    true_count = 100
    for eps in [0.1, 0.5, 1.0, 2.0]:
        noisy = laplace_mechanism(true_count, sensitivity=1, epsilon=eps)
        print(f"   ε={eps}: True={true_count}, Noisy={noisy:.1f}")

    # Demo: Private mean
    print("\n2. Private Vector Mean")
    np.random.seed(42)

    pvm = PrivateVectorMean(dimension=10, epsilon=1.0)

    for _ in range(100):
        vec = np.random.randn(10)
        pvm.add(vec)

    private_mean = pvm.get_private_mean()
    true_mean = pvm.sum / pvm.count

    error = np.linalg.norm(private_mean - true_mean)
    print(f"   Contributors: {pvm.count}")
    print(f"   Error (L2): {error:.4f}")

    # Demo: Exponential mechanism
    print("\n3. Exponential Mechanism (Private Selection)")
    candidates = ['Option A', 'Option B', 'Option C', 'Option D']
    utilities = {'Option A': 10, 'Option B': 8, 'Option C': 5, 'Option D': 2}

    selection_counts = {c: 0 for c in candidates}
    for _ in range(1000):
        selected = exponential_mechanism(
            candidates,
            lambda x: utilities[x],
            sensitivity=1,
            epsilon=1.0
        )
        selection_counts[selected] += 1

    print("   Utility → Selection Frequency:")
    for c in candidates:
        print(f"   {c} (utility={utilities[c]}): {selection_counts[c]/10:.1f}%")

    # Demo: Privacy budget tracking
    print("\n4. Privacy Budget Tracking")
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)

    for i in range(5):
        result = dp.query(100, sensitivity=1, epsilon=0.15)
        print(f"   Query {i+1}: Result={result:.1f}, "
              f"Remaining budget={dp.remaining_budget:.3f}")

    print(f"\n   Total ε used: {dp.budget.epsilon:.3f}")
    print(f"   Advanced composition ε: {dp.budget.advanced_composition():.3f}")

    print("\n" + "=" * 50)
    print("Differential Privacy module ready!")
