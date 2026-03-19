"""
Evolution Strategies — Derivative-Free Optimization via Natural Selection
=========================================================================

Implements three evolution strategy variants for black-box optimization:

    1. CMA-ES (Covariance Matrix Adaptation): The gold standard for
       derivative-free optimization. Adapts a full covariance matrix
       to learn the local landscape geometry.

    2. Natural Evolution Strategy (NES): OpenAI-ES style fitness-weighted
       perturbations using the log-derivative trick.

    3. Simple (1+lambda) ES: Mutation-only baseline for fast prototyping.

Mathematical Foundation:
    CMA-ES:
        Sample:     x_k ~ N(m, sigma^2 * C)
        Rank:       sort by fitness
        Update m:   m += c_m * sum(w_i * (x_{i:lambda} - m))
        Update C:   via evolution path p_c and rank-mu update
        Update sigma: via cumulation path p_sigma (CSA)

    NES:
        Sample:     epsilon_k ~ N(0, I)
        Evaluate:   f(theta + sigma * epsilon_k)
        Gradient:   nabla J ~ (1/n*sigma) * sum(f_k * epsilon_k)
        Update:     theta += lr * nabla J

Pure numpy, no framework dependencies.

Author: Claude Code (with HoloLoom architecture by Blake)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class CMAESState:
    """Internal state of the CMA-ES optimizer."""
    mean: np.ndarray                           # Distribution mean
    sigma: float                               # Step size
    C: np.ndarray                              # Covariance matrix
    p_sigma: np.ndarray                        # Cumulation path for sigma
    p_c: np.ndarray                            # Evolution path for C
    generation: int = 0


@dataclass
class ESResult:
    """Result of an evolution strategy optimization run."""
    best: np.ndarray                           # Best solution found
    best_fitness: float                        # Fitness of best solution
    generations: int                           # Number of generations run
    convergence_curve: list[float] = field(default_factory=list)


# ============================================================================
# CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
# ============================================================================

class CMAES:
    """
    CMA-ES: the state-of-the-art derivative-free optimizer.

    Adapts a multivariate normal distribution N(m, sigma^2 * C)
    to fit the fitness landscape. The covariance matrix C learns
    the local geometry — correlations, scaling, rotation — so the
    search becomes increasingly efficient.

    Key adaptation mechanisms:
    - Rank-mu update: learn from the mu best solutions each generation
    - Rank-one update: accumulate a direction of progress (evolution path)
    - Cumulative step-size adaptation (CSA): control sigma via path length

    Conventions: MINIMIZATION. Lower fitness is better.
    """

    def __init__(
        self,
        dim: int,
        initial_mean: np.ndarray | None = None,
        sigma: float = 0.5,
    ):
        """
        Initialize CMA-ES.

        Args:
            dim: Problem dimensionality
            initial_mean: Starting point (zeros if None)
            sigma: Initial step size
        """
        self.dim = dim

        # Population size (Hansen's default: 4 + floor(3*ln(n)))
        self.lam = 4 + int(3 * np.log(dim))
        self.mu = self.lam // 2

        # Recombination weights (log-linear, normalized)
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_weights / raw_weights.sum()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        # Learning rates
        self.c_sigma = (self.mu_eff + 2) / (dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (dim + 1)) - 1) + self.c_sigma

        self.c_c = (4 + self.mu_eff / dim) / (dim + 4 + 2 * self.mu_eff / dim)
        self.c_1 = 2 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1 - self.c_1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((dim + 2) ** 2 + self.mu_eff),
        )

        # Expected length of N(0,I) vector
        self.chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State
        mean = initial_mean if initial_mean is not None else np.zeros(dim)
        self.state = CMAESState(
            mean=np.asarray(mean, dtype=np.float64).copy(),
            sigma=sigma,
            C=np.eye(dim, dtype=np.float64),
            p_sigma=np.zeros(dim, dtype=np.float64),
            p_c=np.zeros(dim, dtype=np.float64),
            generation=0,
        )

    def ask(self, n_samples: int | None = None) -> np.ndarray:
        """
        Sample candidate solutions from N(mean, sigma^2 * C).

        Args:
            n_samples: Number of samples (defaults to lambda)

        Returns:
            Solutions array (n_samples, dim)
        """
        n = n_samples if n_samples is not None else self.lam

        # Decompose C = B * D^2 * B^T
        try:
            eigenvalues, B = np.linalg.eigh(self.state.C)
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            D = np.sqrt(eigenvalues)
        except np.linalg.LinAlgError:
            B = np.eye(self.dim)
            D = np.ones(self.dim)

        # Sample: x = mean + sigma * B * D * z, where z ~ N(0, I)
        z = np.random.randn(n, self.dim)
        solutions = self.state.mean + self.state.sigma * (z @ np.diag(D) @ B.T)

        return solutions

    def tell(self, solutions: np.ndarray, fitnesses: np.ndarray) -> None:
        """
        Update distribution parameters from evaluated solutions.

        Solutions are ranked by fitness (minimization), and the mu best
        are used to update mean, evolution paths, covariance, and step size.

        Args:
            solutions: Candidate solutions (n, dim)
            fitnesses: Fitness values (n,) — lower is better
        """
        solutions = np.asarray(solutions, dtype=np.float64)
        fitnesses = np.asarray(fitnesses, dtype=np.float64)

        # Sort by fitness (ascending — we minimize)
        order = np.argsort(fitnesses)
        selected = solutions[order[:self.mu]]

        # Old mean
        old_mean = self.state.mean.copy()

        # Update mean: weighted recombination of mu best
        self.state.mean = np.sum(self.weights[:, None] * selected, axis=0)

        # Decompose C for transformation
        try:
            eigenvalues, B = np.linalg.eigh(self.state.C)
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            D = np.sqrt(eigenvalues)
            invsqrt_C = B @ np.diag(1.0 / D) @ B.T
        except np.linalg.LinAlgError:
            invsqrt_C = np.eye(self.dim)

        # Update cumulation path for sigma (CSA)
        self.state.p_sigma = (
            (1 - self.c_sigma) * self.state.p_sigma
            + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff)
            * invsqrt_C @ (self.state.mean - old_mean) / self.state.sigma
        )

        # Heaviside function for stalling detection
        h_sigma = int(
            np.linalg.norm(self.state.p_sigma)
            / np.sqrt(1 - (1 - self.c_sigma) ** (2 * (self.state.generation + 1)))
            < (1.4 + 2 / (self.dim + 1)) * self.chi_n
        )

        # Update evolution path for C
        self.state.p_c = (
            (1 - self.c_c) * self.state.p_c
            + h_sigma
            * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff)
            * (self.state.mean - old_mean) / self.state.sigma
        )

        # Rank-one update component
        rank_one = np.outer(self.state.p_c, self.state.p_c)

        # Rank-mu update component
        steps = (selected - old_mean) / self.state.sigma
        rank_mu = np.sum(
            self.weights[k] * np.outer(steps[k], steps[k])
            for k in range(self.mu)
        )

        # Update covariance matrix
        self.state.C = (
            (1 - self.c_1 - self.c_mu) * self.state.C
            + self.c_1 * (
                rank_one
                + (1 - h_sigma) * self.c_c * (2 - self.c_c) * self.state.C
            )
            + self.c_mu * rank_mu
        )

        # Enforce symmetry
        self.state.C = (self.state.C + self.state.C.T) / 2

        # Update step size via CSA
        self.state.sigma *= np.exp(
            (self.c_sigma / self.d_sigma)
            * (np.linalg.norm(self.state.p_sigma) / self.chi_n - 1)
        )

        self.state.generation += 1

    def optimize(
        self,
        f: Callable[[np.ndarray], float],
        n_generations: int = 100,
        tol: float = 1e-12,
    ) -> ESResult:
        """
        Full CMA-ES optimization loop.

        Args:
            f: Objective function to MINIMIZE (solution -> scalar)
            n_generations: Maximum generations
            tol: Stop if sigma drops below this

        Returns:
            ESResult with best solution, fitness, and convergence curve
        """
        best = self.state.mean.copy()
        best_fitness = f(best)
        convergence = [best_fitness]

        for _ in range(n_generations):
            solutions = self.ask()
            fitnesses = np.array([f(s) for s in solutions])

            self.tell(solutions, fitnesses)

            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < best_fitness:
                best = solutions[gen_best_idx].copy()
                best_fitness = fitnesses[gen_best_idx]

            convergence.append(best_fitness)

            if self.state.sigma < tol:
                break

        return ESResult(
            best=best,
            best_fitness=best_fitness,
            generations=self.state.generation,
            convergence_curve=convergence,
        )


# ============================================================================
# Natural Evolution Strategy (OpenAI-ES style)
# ============================================================================

class NaturalEvolutionStrategy:
    """
    Natural Evolution Strategy using fitness-weighted perturbations.

    Instead of ranking and selecting, NES estimates the gradient of
    expected fitness using the log-derivative trick:

        nabla_theta E[f(theta + sigma*eps)] ~ (1/n*sigma) * sum(f_i * eps_i)

    This is the approach used by OpenAI's evolution strategy paper.
    Simple, parallelizable, and surprisingly effective.

    Convention: MAXIMIZATION. Higher fitness is better.
    """

    def optimize(
        self,
        f: Callable[[np.ndarray], float],
        initial: np.ndarray,
        sigma: float = 0.1,
        lr: float = 0.01,
        n_generations: int = 100,
        pop_size: int = 50,
    ) -> ESResult:
        """
        Run NES optimization.

        Args:
            f: Objective function to MAXIMIZE
            initial: Starting point
            sigma: Perturbation scale
            lr: Learning rate
            n_generations: Number of generations
            pop_size: Population size (should be even for antithetic sampling)

        Returns:
            ESResult with best solution and convergence curve
        """
        theta = np.asarray(initial, dtype=np.float64).copy()
        dim = len(theta)

        best = theta.copy()
        best_fitness = f(theta)
        convergence = [best_fitness]

        # Use antithetic sampling (mirror perturbations) for variance reduction
        half_pop = pop_size // 2

        for gen in range(n_generations):
            # Sample perturbations
            eps = np.random.randn(half_pop, dim)
            # Mirror: use both +eps and -eps
            eps_full = np.vstack([eps, -eps])

            # Evaluate
            fitnesses = np.array([
                f(theta + sigma * eps_full[i])
                for i in range(len(eps_full))
            ])

            # Normalize fitnesses (rank-based or z-score)
            f_mean = fitnesses.mean()
            f_std = fitnesses.std()
            if f_std > 1e-12:
                fitnesses_norm = (fitnesses - f_mean) / f_std
            else:
                fitnesses_norm = fitnesses - f_mean

            # Gradient estimate: (1/n*sigma) * sum(f_i * eps_i)
            grad = (1.0 / (len(eps_full) * sigma)) * (fitnesses_norm @ eps_full)

            # Gradient ascent (maximization)
            theta += lr * grad

            current_fitness = f(theta)
            if current_fitness > best_fitness:
                best = theta.copy()
                best_fitness = current_fitness

            convergence.append(best_fitness)

        return ESResult(
            best=best,
            best_fitness=best_fitness,
            generations=n_generations,
            convergence_curve=convergence,
        )


# ============================================================================
# Simple (1+lambda) ES
# ============================================================================

class SimpleES:
    """
    Simple (1+lambda) evolution strategy with Gaussian mutation.

    The simplest possible ES: generate lambda mutants, keep the best
    (including the parent). No covariance adaptation, no cumulation,
    no recombination. Useful as a baseline or for low-dimensional
    problems where CMA-ES overhead isn't warranted.

    Convention: MINIMIZATION. Lower fitness is better.
    """

    def optimize(
        self,
        f: Callable[[np.ndarray], float],
        dim: int,
        initial: np.ndarray | None = None,
        sigma: float = 0.1,
        pop_size: int = 20,
        n_generations: int = 100,
        sigma_decay: float = 1.0,
    ) -> ESResult:
        """
        Run (1+lambda) ES.

        Args:
            f: Objective function to MINIMIZE
            dim: Problem dimensionality
            initial: Starting point (random if None)
            sigma: Mutation standard deviation
            pop_size: Lambda (number of offspring per generation)
            n_generations: Maximum generations
            sigma_decay: Multiply sigma by this each generation (< 1 = annealing)

        Returns:
            ESResult with best solution and convergence curve
        """
        if initial is not None:
            parent = np.asarray(initial, dtype=np.float64).copy()
        else:
            parent = np.random.randn(dim)

        parent_fitness = f(parent)
        best = parent.copy()
        best_fitness = parent_fitness
        convergence = [best_fitness]

        current_sigma = sigma

        for gen in range(n_generations):
            # Generate offspring by Gaussian mutation
            offspring = parent + current_sigma * np.random.randn(pop_size, dim)

            # Evaluate
            offspring_fitnesses = np.array([f(o) for o in offspring])

            # Select best among parent + offspring
            best_offspring_idx = np.argmin(offspring_fitnesses)

            if offspring_fitnesses[best_offspring_idx] < parent_fitness:
                parent = offspring[best_offspring_idx].copy()
                parent_fitness = offspring_fitnesses[best_offspring_idx]

            if parent_fitness < best_fitness:
                best = parent.copy()
                best_fitness = parent_fitness

            convergence.append(best_fitness)
            current_sigma *= sigma_decay

        return ESResult(
            best=best,
            best_fitness=best_fitness,
            generations=n_generations,
            convergence_curve=convergence,
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "CMAESState",
    "ESResult",
    "CMAES",
    "NaturalEvolutionStrategy",
    "SimpleES",
]
