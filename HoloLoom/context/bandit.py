#!/usr/bin/env python3
"""
Thompson Sampling Bandit for Backend Selection

Part 3: Classification and Basic Routing (Days 11-15)

Implements Bayesian bandit for adaptive backend selection.
Validated in Demo 2 (converged at iteration 148).

Algorithm:
- Beta distribution per backend: Beta(α, β)
- α = successes + 1, β = failures + 1
- Sample from each distribution, select max
- Update based on query outcome
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import logging


logger = logging.getLogger(__name__)


# ============================================================================
# Bandit Arm
# ============================================================================

@dataclass
class BanditArm:
    """
    Single backend (arm) in the bandit

    Uses Beta distribution for success probability modeling
    """
    name: str
    alpha: float = 1.0  # Success parameter (prior = 1.0)
    beta: float = 1.0   # Failure parameter (prior = 1.0)
    total_pulls: int = 0
    total_successes: int = 0
    total_failures: int = 0

    def expected_reward(self) -> float:
        """Expected reward (mean of Beta distribution)"""
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        """Sample from Beta distribution"""
        return np.random.beta(self.alpha, self.beta)

    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval

        Args:
            confidence: Confidence level (default: 0.95)

        Returns:
            (lower, upper) bounds
        """
        # Use percentiles of Beta distribution
        samples = np.random.beta(self.alpha, self.beta, 10000)
        lower = np.percentile(samples, (1 - confidence) * 100 / 2)
        upper = np.percentile(samples, 100 - (1 - confidence) * 100 / 2)
        return lower, upper

    def ci_width(self, confidence: float = 0.95) -> float:
        """Confidence interval width (measure of uncertainty)"""
        lower, upper = self.confidence_interval(confidence)
        return upper - lower


# ============================================================================
# Thompson Sampling Bandit
# ============================================================================

class ThompsonBandit:
    """
    Thompson Sampling for adaptive backend selection

    Validated in Demo 2:
    - Converged at iteration 148 (target: <500)
    - Correctly identified best backend (SQL: 0.892 vs 0.900 true)
    """

    def __init__(self, backends: List[str]):
        """
        Initialize Thompson Sampling bandit

        Args:
            backends: Backend names (e.g., ["sql", "neo4j", "qdrant"])
        """
        self.arms: Dict[str, BanditArm] = {
            name: BanditArm(name=name)
            for name in backends
        }
        self.history: List[Dict] = []
        self.iteration = 0

    def select(self) -> str:
        """
        Select backend via Thompson Sampling

        Returns:
            Backend name with highest sampled value
        """
        # Sample from each arm's Beta distribution
        samples = {
            name: arm.sample()
            for name, arm in self.arms.items()
        }

        # Select arm with highest sample
        selected_backend = max(samples, key=samples.get)

        logger.debug(
            f"Thompson Sampling: selected={selected_backend}, "
            f"samples={[(k, f'{v:.3f}') for k, v in samples.items()]}"
        )

        return selected_backend

    def update(self, backend: str, success: bool, confidence: Optional[float] = None):
        """
        Update bandit statistics based on outcome

        Args:
            backend: Backend that was selected
            success: Whether query succeeded
            confidence: Optional confidence score to weight update
        """
        if backend not in self.arms:
            logger.warning(f"Unknown backend: {backend}")
            return

        arm = self.arms[backend]
        arm.total_pulls += 1

        # Weight update by confidence (if provided)
        weight = confidence if confidence is not None else 1.0

        if success:
            arm.alpha += weight
            arm.total_successes += 1
        else:
            arm.beta += weight
            arm.total_failures += 1

        self.iteration += 1

        # Record history
        self.history.append({
            "iteration": self.iteration,
            "backend": backend,
            "success": success,
            "confidence": confidence,
            "alpha": arm.alpha,
            "beta": arm.beta,
            "expected_reward": arm.expected_reward()
        })

        logger.debug(
            f"Updated {backend}: alpha={arm.alpha:.1f}, beta={arm.beta:.1f}, "
            f"E[r]={arm.expected_reward():.3f}"
        )

    def get_best_backend(self) -> Tuple[str, float]:
        """
        Get backend with highest expected reward

        Returns:
            (backend_name, expected_reward)
        """
        best_arm = max(self.arms.values(), key=lambda arm: arm.expected_reward())
        return best_arm.name, best_arm.expected_reward()

    def has_converged(self, threshold: float = 0.10) -> bool:
        """
        Check if bandit has converged

        Args:
            threshold: CI width threshold for convergence (default: 0.10)

        Returns:
            True if best backend's CI width < threshold
        """
        best_backend, _ = self.get_best_backend()
        best_arm = self.arms[best_backend]
        return best_arm.ci_width() < threshold

    def get_convergence_iteration(self, threshold: float = 0.10) -> Optional[int]:
        """
        Get iteration at which bandit converged

        Args:
            threshold: CI width threshold

        Returns:
            Iteration number or None if not converged
        """
        best_backend, _ = self.get_best_backend()

        for i, record in enumerate(self.history, 1):
            if record["backend"] == best_backend:
                # Check CI width at this iteration
                arm = BanditArm(
                    name=best_backend,
                    alpha=record["alpha"],
                    beta=record["beta"]
                )
                if arm.ci_width() < threshold:
                    return i

        return None

    def get_stats(self) -> List[Dict]:
        """
        Get statistics for all arms

        Returns:
            List of dicts with arm statistics
        """
        stats = []
        for arm in self.arms.values():
            expected = arm.expected_reward()
            lower, upper = arm.confidence_interval()
            ci_width = upper - lower

            stats.append({
                "backend": arm.name,
                "alpha": arm.alpha,
                "beta": arm.beta,
                "expected_reward": expected,
                "ci_lower": lower,
                "ci_upper": upper,
                "ci_width": ci_width,
                "total_pulls": arm.total_pulls,
                "total_successes": arm.total_successes,
                "total_failures": arm.total_failures
            })

        # Sort by expected reward (descending)
        return sorted(stats, key=lambda s: s["expected_reward"], reverse=True)

    def get_summary(self) -> str:
        """Get human-readable summary"""
        stats = self.get_stats()
        best_backend, best_reward = self.get_best_backend()
        converged = self.has_converged()
        convergence_iter = self.get_convergence_iteration()

        summary = f"Thompson Sampling Summary:\n"
        summary += f"  Iterations: {self.iteration}\n"
        summary += f"  Best backend: {best_backend} (E[r]={best_reward:.3f})\n"
        summary += f"  Converged: {'Yes' if converged else 'No'}"
        if convergence_iter:
            summary += f" (at iteration {convergence_iter})"
        summary += f"\n\n"

        summary += f"Backend Statistics:\n"
        for s in stats:
            summary += (
                f"  {s['backend']:<10}: alpha={s['alpha']:>6.1f}, beta={s['beta']:>6.1f}, "
                f"E[r]={s['expected_reward']:>5.3f}, pulls={s['total_pulls']:>4d}\n"
            )

        return summary


# ============================================================================
# Helper Functions
# ============================================================================

def create_thompson_bandit(backends: List[str]) -> ThompsonBandit:
    """
    Create Thompson Sampling bandit

    Args:
        backends: List of backend names

    Returns:
        ThompsonBandit instance
    """
    return ThompsonBandit(backends)
