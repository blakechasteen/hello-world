"""
Genetic Algorithms — Evolutionary Optimization
================================================

Population-based optimization inspired by natural selection.
Individuals with higher fitness reproduce, creating new
candidates via crossover and mutation.

Core Concepts:
- Selection: Tournament, roulette wheel
- Crossover: Single-point, uniform
- Mutation: Gaussian perturbation
- Differential Evolution: Mutation via vector differences

Mathematical Foundation:
GA: Select parents ∝ fitness, recombine genes, mutate
DE: v = x_r1 + F(x_r2 - x_r3), trial = crossover(v, x_i)

Applications to Warp Space:
- Hyperparameter optimization for scoring terms
- Evolving Thompson Sampling strategies
- Architecture search for embedding networks

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Individual:
    """A single candidate solution.

    Attributes:
        genome: Parameter vector (genome_length,)
        fitness: Fitness value (higher is better)
    """
    genome: np.ndarray
    fitness: float = -np.inf


# ============================================================================
# SELECTION METHODS
# ============================================================================

def _tournament_selection(
    population: list[Individual],
    tournament_size: int = 3,
) -> Individual:
    """Tournament selection: pick best from random subset."""
    contestants = np.random.choice(len(population), size=tournament_size, replace=False)
    best = max(contestants, key=lambda i: population[i].fitness)
    return population[best]


def _roulette_selection(population: list[Individual]) -> Individual:
    """Roulette wheel (fitness-proportionate) selection."""
    fitnesses = np.array([ind.fitness for ind in population])

    # Shift to positive
    min_f = fitnesses.min()
    if min_f < 0:
        fitnesses = fitnesses - min_f + 1e-6

    total = fitnesses.sum()
    if total < 1e-15:
        return population[np.random.randint(len(population))]

    probs = fitnesses / total
    idx = np.random.choice(len(population), p=probs)
    return population[idx]


# ============================================================================
# CROSSOVER METHODS
# ============================================================================

def _single_point_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-point crossover."""
    n = len(parent1)
    point = np.random.randint(1, n)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


def _uniform_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    swap_prob: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform crossover: each gene independently from either parent."""
    mask = np.random.random(len(parent1)) < swap_prob
    child1 = np.where(mask, parent2, parent1)
    child2 = np.where(mask, parent1, parent2)
    return child1, child2


# ============================================================================
# GENETIC ALGORITHM
# ============================================================================

class GeneticAlgorithm:
    """Standard Genetic Algorithm for optimization.

    Maintains a population of candidate solutions. Each generation:
    1. Evaluate fitness
    2. Select parents (tournament or roulette)
    3. Crossover to produce offspring
    4. Mutate offspring
    5. Replace population (elitism preserves best)

    Example:
        >>> ga = GeneticAlgorithm()
        >>> def sphere(x): return -np.sum(x**2)  # Maximize = minimize sum of squares
        >>> best = ga.evolve(sphere, pop_size=50, genome_length=10,
        ...                  n_generations=100)
        >>> np.allclose(best.genome, 0, atol=0.5)
        True
    """

    def evolve(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        pop_size: int = 50,
        genome_length: int = 10,
        n_generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.1,
        selection: str = "tournament",
        crossover: str = "single_point",
        tournament_size: int = 3,
        elitism: int = 1,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> Individual:
        """Run genetic algorithm.

        Args:
            fitness_fn: Function mapping genome -> fitness (maximize)
            pop_size: Population size
            genome_length: Length of genome vector
            n_generations: Number of generations
            crossover_rate: Probability of crossover vs cloning
            mutation_rate: Probability of mutating each gene
            mutation_scale: Std dev of Gaussian mutation
            selection: "tournament" or "roulette"
            crossover: "single_point" or "uniform"
            tournament_size: Tournament size (if tournament selection)
            elitism: Number of best individuals to preserve
            bounds: (lower, upper) bounds for genome values

        Returns:
            Best Individual found
        """
        # Initialize population
        if bounds is not None:
            lo, hi = np.asarray(bounds[0]), np.asarray(bounds[1])
            population = [
                Individual(genome=np.random.uniform(lo, hi))
                for _ in range(pop_size)
            ]
        else:
            population = [
                Individual(genome=np.random.randn(genome_length))
                for _ in range(pop_size)
            ]

        # Evaluate initial population
        for ind in population:
            ind.fitness = fitness_fn(ind.genome)

        best_ever = max(population, key=lambda i: i.fitness)

        select_fn = (
            lambda pop: _tournament_selection(pop, tournament_size)
            if selection == "tournament"
            else lambda pop: _roulette_selection(pop)
        )

        cross_fn = (
            _single_point_crossover if crossover == "single_point"
            else _uniform_crossover
        )

        for gen in range(n_generations):
            # Sort by fitness (descending)
            population.sort(key=lambda i: i.fitness, reverse=True)

            # Elite carry-over
            new_pop = [Individual(genome=population[i].genome.copy(),
                                  fitness=population[i].fitness)
                       for i in range(min(elitism, pop_size))]

            # Fill rest with offspring
            while len(new_pop) < pop_size:
                parent1 = select_fn(population)
                parent2 = select_fn(population)

                if np.random.random() < crossover_rate:
                    g1, g2 = cross_fn(parent1.genome, parent2.genome)
                else:
                    g1 = parent1.genome.copy()
                    g2 = parent2.genome.copy()

                # Mutation
                for g in [g1, g2]:
                    mask = np.random.random(len(g)) < mutation_rate
                    g[mask] += np.random.randn(np.sum(mask)) * mutation_scale

                    if bounds is not None:
                        g[:] = np.clip(g, lo, hi)

                new_pop.append(Individual(genome=g1, fitness=fitness_fn(g1)))
                if len(new_pop) < pop_size:
                    new_pop.append(Individual(genome=g2, fitness=fitness_fn(g2)))

            population = new_pop[:pop_size]

            gen_best = max(population, key=lambda i: i.fitness)
            if gen_best.fitness > best_ever.fitness:
                best_ever = Individual(
                    genome=gen_best.genome.copy(),
                    fitness=gen_best.fitness,
                )

        return best_ever


# ============================================================================
# DIFFERENTIAL EVOLUTION
# ============================================================================

class DifferentialEvolution:
    """Differential Evolution — mutation via vector differences.

    Simple, robust, global optimizer. No gradient needed.

    Mutation: v_i = x_r1 + F * (x_r2 - x_r3)
    Crossover: trial_j = v_j if rand < CR else x_ij
    Selection: keep trial if better than parent

    Example:
        >>> de = DifferentialEvolution()
        >>> def rosenbrock(x):
        ...     return -sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2
        ...                 for i in range(len(x)-1))
        >>> best = de.optimize(rosenbrock, bounds=[(-5, 5)]*5,
        ...                    pop_size=50, F=0.8, CR=0.9,
        ...                    n_generations=500)
        >>> np.allclose(best.genome, 1, atol=0.5)
        True
    """

    def optimize(
        self,
        f: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        pop_size: int = 50,
        F: float = 0.8,
        CR: float = 0.9,
        n_generations: int = 200,
        strategy: str = "rand1bin",
    ) -> Individual:
        """Run Differential Evolution.

        Args:
            f: Fitness function (maximize)
            bounds: [(lo, hi)] for each dimension
            pop_size: Population size (≥ 4)
            F: Differential weight (scale factor), typically [0.4, 1.0]
            CR: Crossover probability [0, 1]
            n_generations: Number of generations
            strategy: Mutation strategy ("rand1bin" or "best1bin")

        Returns:
            Best Individual found
        """
        bounds = np.array(bounds, dtype=float)
        lo = bounds[:, 0]
        hi = bounds[:, 1]
        d = len(bounds)

        # Initialize population
        pop = np.random.uniform(lo, hi, size=(pop_size, d))
        fitness = np.array([f(x) for x in pop])

        best_idx = np.argmax(fitness)
        best = Individual(genome=pop[best_idx].copy(), fitness=fitness[best_idx])

        for gen in range(n_generations):
            for i in range(pop_size):
                # Select 3 distinct random individuals ≠ i
                candidates = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

                if strategy == "best1bin":
                    base = pop[best_idx]
                else:
                    base = pop[r1]

                # Mutation
                mutant = base + F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lo, hi)

                # Binomial crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(d)  # Ensure at least one from mutant
                for j in range(d):
                    if np.random.random() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Selection
                trial_fitness = f(trial)
                if trial_fitness >= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness > best.fitness:
                        best = Individual(genome=trial.copy(), fitness=trial_fitness)
                        best_idx = i

        return best
