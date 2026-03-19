"""
Dark Trace Research: Genetic Adversarial Search
================================================

Gradient-free adversarial probing using genetic algorithms.
Useful for black-box models or non-differentiable feature functions.

Research Basis:
- Genetic Algorithms for Adversarial Examples (Alzantot et al., 2018)
- One Pixel Attack (Su et al., 2019)
- Natural Evolution Strategies (Wierstra et al., 2014)

Key Capabilities:
- Black-box adversarial search (no gradient required)
- Population-based exploration
- Multi-objective fitness (activation + perturbation)
- Crossover and mutation operators
- Elite preservation

Created: 2025-12-28
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# =============================================================================
# Configuration
# =============================================================================


class SelectionMethod(Enum):
    """Selection method for genetic algorithm."""
    TOURNAMENT = "tournament"       # Tournament selection
    ROULETTE = "roulette"          # Roulette wheel selection
    RANK = "rank"                  # Rank-based selection
    ELITE = "elite"                # Elite selection only


class CrossoverMethod(Enum):
    """Crossover method for genetic algorithm."""
    UNIFORM = "uniform"            # Uniform crossover
    SINGLE_POINT = "single_point"  # Single point crossover
    TWO_POINT = "two_point"        # Two point crossover
    BLEND = "blend"                # BLX-alpha blend crossover


class MutationMethod(Enum):
    """Mutation method for genetic algorithm."""
    GAUSSIAN = "gaussian"          # Gaussian noise mutation
    UNIFORM = "uniform"            # Uniform noise mutation
    ADAPTIVE = "adaptive"          # Adaptive mutation rate
    CAUCHY = "cauchy"              # Cauchy distribution (heavy tails)


@dataclass
class GeneticConfig:
    """
    Configuration for genetic adversarial search.

    Attributes:
        population_size: Number of individuals in population
        generations: Maximum generations to evolve
        mutation_rate: Probability of mutation per gene
        crossover_rate: Probability of crossover
        elite_count: Number of elite individuals to preserve
        tournament_size: Size of tournament for selection
        epsilon: Maximum perturbation bound (L-inf)
        l2_bound: Optional L2 norm bound
        fitness_weights: Weights for multi-objective fitness
        selection_method: Method for parent selection
        crossover_method: Method for crossover
        mutation_method: Method for mutation
        adaptive_mutation: Enable adaptive mutation rate
        early_stopping_patience: Generations without improvement before stopping
        target_fitness: Stop if this fitness is achieved
        seed: Random seed for reproducibility
    """
    population_size: int = 50
    generations: int = 30
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_count: int = 5
    tournament_size: int = 5
    epsilon: float = 0.1
    l2_bound: float | None = None
    fitness_weights: dict[str, float] = field(default_factory=lambda: {
        "activation_change": 0.7,
        "perturbation_size": -0.3  # Negative = minimize
    })
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.BLEND
    mutation_method: MutationMethod = MutationMethod.ADAPTIVE
    adaptive_mutation: bool = True
    early_stopping_patience: int = 10
    target_fitness: float | None = None
    seed: int | None = None

    @classmethod
    def fast(cls) -> "GeneticConfig":
        """Fast configuration for quick probing."""
        return cls(
            population_size=20,
            generations=15,
            mutation_rate=0.2,
            elite_count=2,
            tournament_size=3,
            early_stopping_patience=5
        )

    @classmethod
    def balanced(cls) -> "GeneticConfig":
        """Balanced configuration (default)."""
        return cls()

    @classmethod
    def thorough(cls) -> "GeneticConfig":
        """Thorough configuration for comprehensive search."""
        return cls(
            population_size=100,
            generations=100,
            mutation_rate=0.05,
            crossover_rate=0.9,
            elite_count=10,
            tournament_size=7,
            early_stopping_patience=20
        )


# =============================================================================
# Individual Representation
# =============================================================================


@dataclass
class Individual:
    """
    Individual in the genetic algorithm population.

    Represents a perturbation vector that can be applied to input.
    """
    genes: np.ndarray              # Perturbation vector
    fitness: float = 0.0           # Fitness score
    activation_change: float = 0.0 # How much activation changed
    perturbation_norm: float = 0.0 # L2 norm of perturbation
    generation: int = 0            # Generation when created
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "Individual":
        """Create a copy of this individual."""
        return Individual(
            genes=self.genes.copy(),
            fitness=self.fitness,
            activation_change=self.activation_change,
            perturbation_norm=self.perturbation_norm,
            generation=self.generation,
            metadata=self.metadata.copy()
        )


@dataclass
class EvolutionResult:
    """Result of genetic evolution."""
    best_individual: Individual
    final_population: list[Individual]
    generations_run: int
    fitness_history: list[float]     # Best fitness per generation
    diversity_history: list[float]   # Population diversity per generation
    converged: bool                  # Whether search converged
    early_stopped: bool              # Whether early stopping triggered
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Genetic Operators
# =============================================================================


class GeneticOperators:
    """
    Genetic operators for adversarial search.

    Provides selection, crossover, and mutation operations.
    """

    def __init__(self, config: GeneticConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._mutation_rate = config.mutation_rate

    # -------------------------------------------------------------------------
    # Selection
    # -------------------------------------------------------------------------

    def select_parents(
        self,
        population: list[Individual],
        n_parents: int
    ) -> list[Individual]:
        """Select parents for reproduction."""
        if self.config.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection(population, n_parents)
        elif self.config.selection_method == SelectionMethod.ROULETTE:
            return self._roulette_selection(population, n_parents)
        elif self.config.selection_method == SelectionMethod.RANK:
            return self._rank_selection(population, n_parents)
        elif self.config.selection_method == SelectionMethod.ELITE:
            return self._elite_selection(population, n_parents)
        else:
            return self._tournament_selection(population, n_parents)

    def _tournament_selection(
        self,
        population: list[Individual],
        n_parents: int
    ) -> list[Individual]:
        """Tournament selection."""
        parents = []
        for _ in range(n_parents):
            # Random tournament
            tournament = self.rng.choice(
                population,
                size=min(self.config.tournament_size, len(population)),
                replace=False
            )
            # Select best from tournament
            winner = max(tournament, key=lambda ind: ind.fitness)
            parents.append(winner.copy())
        return parents

    def _roulette_selection(
        self,
        population: list[Individual],
        n_parents: int
    ) -> list[Individual]:
        """Roulette wheel selection."""
        # Shift fitness to be positive
        fitnesses = np.array([ind.fitness for ind in population])
        min_fit = fitnesses.min()
        shifted = fitnesses - min_fit + 1e-6

        # Normalize to probabilities
        probs = shifted / shifted.sum()

        # Select
        indices = self.rng.choice(
            len(population),
            size=n_parents,
            replace=True,
            p=probs
        )
        return [population[i].copy() for i in indices]

    def _rank_selection(
        self,
        population: list[Individual],
        n_parents: int
    ) -> list[Individual]:
        """Rank-based selection."""
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)
        n = len(sorted_pop)

        # Assign rank probabilities (linear)
        ranks = np.arange(1, n + 1)
        probs = ranks / ranks.sum()

        indices = self.rng.choice(n, size=n_parents, replace=True, p=probs)
        return [sorted_pop[i].copy() for i in indices]

    def _elite_selection(
        self,
        population: list[Individual],
        n_parents: int
    ) -> list[Individual]:
        """Elite selection (top individuals only)."""
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        elite = sorted_pop[:max(n_parents, self.config.elite_count)]
        return [ind.copy() for ind in elite[:n_parents]]

    # -------------------------------------------------------------------------
    # Crossover
    # -------------------------------------------------------------------------

    def crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        if self.rng.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()

        if self.config.crossover_method == CrossoverMethod.UNIFORM:
            return self._uniform_crossover(parent1, parent2)
        elif self.config.crossover_method == CrossoverMethod.SINGLE_POINT:
            return self._single_point_crossover(parent1, parent2)
        elif self.config.crossover_method == CrossoverMethod.TWO_POINT:
            return self._two_point_crossover(parent1, parent2)
        elif self.config.crossover_method == CrossoverMethod.BLEND:
            return self._blend_crossover(parent1, parent2)
        else:
            return self._uniform_crossover(parent1, parent2)

    def _uniform_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> tuple[Individual, Individual]:
        """Uniform crossover - each gene randomly from either parent."""
        mask = self.rng.random(parent1.genes.shape) < 0.5

        child1_genes = np.where(mask, parent1.genes, parent2.genes)
        child2_genes = np.where(mask, parent2.genes, parent1.genes)

        return (
            Individual(genes=child1_genes),
            Individual(genes=child2_genes)
        )

    def _single_point_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> tuple[Individual, Individual]:
        """Single point crossover."""
        point = self.rng.integers(1, len(parent1.genes.flat))

        genes1 = parent1.genes.flatten()
        genes2 = parent2.genes.flatten()

        child1_genes = np.concatenate([genes1[:point], genes2[point:]])
        child2_genes = np.concatenate([genes2[:point], genes1[point:]])

        return (
            Individual(genes=child1_genes.reshape(parent1.genes.shape)),
            Individual(genes=child2_genes.reshape(parent2.genes.shape))
        )

    def _two_point_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> tuple[Individual, Individual]:
        """Two point crossover."""
        size = len(parent1.genes.flat)
        point1, point2 = sorted(self.rng.choice(size, size=2, replace=False))

        genes1 = parent1.genes.flatten()
        genes2 = parent2.genes.flatten()

        child1_genes = genes1.copy()
        child2_genes = genes2.copy()

        child1_genes[point1:point2] = genes2[point1:point2]
        child2_genes[point1:point2] = genes1[point1:point2]

        return (
            Individual(genes=child1_genes.reshape(parent1.genes.shape)),
            Individual(genes=child2_genes.reshape(parent2.genes.shape))
        )

    def _blend_crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        alpha: float = 0.5
    ) -> tuple[Individual, Individual]:
        """BLX-alpha blend crossover."""
        diff = np.abs(parent1.genes - parent2.genes)
        low = np.minimum(parent1.genes, parent2.genes) - alpha * diff
        high = np.maximum(parent1.genes, parent2.genes) + alpha * diff

        child1_genes = self.rng.uniform(low, high)
        child2_genes = self.rng.uniform(low, high)

        return (
            Individual(genes=child1_genes),
            Individual(genes=child2_genes)
        )

    # -------------------------------------------------------------------------
    # Mutation
    # -------------------------------------------------------------------------

    def mutate(self, individual: Individual) -> Individual:
        """Mutate an individual."""
        if self.config.mutation_method == MutationMethod.GAUSSIAN:
            return self._gaussian_mutation(individual)
        elif self.config.mutation_method == MutationMethod.UNIFORM:
            return self._uniform_mutation(individual)
        elif self.config.mutation_method == MutationMethod.ADAPTIVE:
            return self._adaptive_mutation(individual)
        elif self.config.mutation_method == MutationMethod.CAUCHY:
            return self._cauchy_mutation(individual)
        else:
            return self._gaussian_mutation(individual)

    def _gaussian_mutation(self, individual: Individual) -> Individual:
        """Gaussian noise mutation."""
        mutated = individual.copy()

        # Mutation mask
        mask = self.rng.random(mutated.genes.shape) < self._mutation_rate

        # Add Gaussian noise
        noise = self.rng.normal(0, self.config.epsilon / 3, mutated.genes.shape)
        mutated.genes = np.where(mask, mutated.genes + noise, mutated.genes)

        # Clip to bounds
        mutated.genes = np.clip(mutated.genes, -self.config.epsilon, self.config.epsilon)

        return mutated

    def _uniform_mutation(self, individual: Individual) -> Individual:
        """Uniform noise mutation."""
        mutated = individual.copy()

        mask = self.rng.random(mutated.genes.shape) < self._mutation_rate
        noise = self.rng.uniform(-self.config.epsilon, self.config.epsilon, mutated.genes.shape)
        mutated.genes = np.where(mask, mutated.genes + noise, mutated.genes)
        mutated.genes = np.clip(mutated.genes, -self.config.epsilon, self.config.epsilon)

        return mutated

    def _adaptive_mutation(self, individual: Individual) -> Individual:
        """Adaptive mutation - rate decreases with fitness."""
        mutated = individual.copy()

        # Adapt mutation rate based on fitness (higher fitness = lower mutation)
        fitness_factor = 1.0 / (1.0 + individual.fitness)
        adaptive_rate = self._mutation_rate * (0.5 + fitness_factor)

        mask = self.rng.random(mutated.genes.shape) < adaptive_rate
        noise = self.rng.normal(0, self.config.epsilon / 3, mutated.genes.shape)
        mutated.genes = np.where(mask, mutated.genes + noise, mutated.genes)
        mutated.genes = np.clip(mutated.genes, -self.config.epsilon, self.config.epsilon)

        return mutated

    def _cauchy_mutation(self, individual: Individual) -> Individual:
        """Cauchy distribution mutation (heavy tails for exploration)."""
        mutated = individual.copy()

        mask = self.rng.random(mutated.genes.shape) < self._mutation_rate

        # Cauchy distribution via inverse CDF
        u = self.rng.random(mutated.genes.shape)
        noise = self.config.epsilon / 3 * np.tan(np.pi * (u - 0.5))

        mutated.genes = np.where(mask, mutated.genes + noise, mutated.genes)
        mutated.genes = np.clip(mutated.genes, -self.config.epsilon, self.config.epsilon)

        return mutated

    def update_mutation_rate(self, generation: int, best_fitness_history: list[float]):
        """Update mutation rate adaptively based on progress."""
        if not self.config.adaptive_mutation:
            return

        # Check if stuck (no improvement)
        if len(best_fitness_history) >= 5:
            recent = best_fitness_history[-5:]
            if max(recent) - min(recent) < 1e-4:
                # Increase mutation to escape local optima
                self._mutation_rate = min(0.5, self._mutation_rate * 1.5)
            else:
                # Decrease mutation for fine-tuning
                self._mutation_rate = max(0.01, self._mutation_rate * 0.95)


# =============================================================================
# Main Genetic Search
# =============================================================================


class GeneticAdversarialSearch:
    """
    Genetic algorithm for adversarial example search.

    Black-box optimization approach that doesn't require gradients.
    Useful for:
    - Non-differentiable activation functions
    - Black-box models (API access only)
    - Discrete perturbations
    - Multi-objective optimization

    Usage:
        config = GeneticConfig.balanced()
        search = GeneticAdversarialSearch(config)

        result = search.evolve(
            input_shape=(1, 384),
            fitness_fn=lambda pert: compute_fitness(original_input + pert),
            target_activation=0.5  # Optional target
        )

        adversarial = original_input + result.best_individual.genes
    """

    def __init__(self, config: GeneticConfig):
        """
        Initialize genetic search.

        Args:
            config: Genetic algorithm configuration
        """
        self.config = config
        self.operators = GeneticOperators(config)
        self.rng = np.random.default_rng(config.seed)

    def evolve(
        self,
        input_shape: tuple[int, ...],
        fitness_fn: Callable[[np.ndarray], float],
        target_activation: float | None = None,
        initial_population: list[Individual] | None = None
    ) -> EvolutionResult:
        """
        Run genetic evolution to find adversarial perturbation.

        Args:
            input_shape: Shape of input to perturb
            fitness_fn: Function(perturbation) -> fitness score
            target_activation: Optional target activation level
            initial_population: Optional starting population

        Returns:
            EvolutionResult with best individual and history
        """
        # Initialize population
        if initial_population is not None:
            population = initial_population
        else:
            population = self._initialize_population(input_shape)

        # Evaluate initial fitness
        self._evaluate_population(population, fitness_fn)

        # Track history
        fitness_history = []
        diversity_history = []
        best_fitness = float('-inf')
        patience_counter = 0

        # Evolution loop
        generations_run = 0
        converged = False
        early_stopped = False

        for gen in range(self.config.generations):
            generations_run = gen + 1

            # Sort by fitness
            population.sort(key=lambda ind: ind.fitness, reverse=True)

            # Track best
            current_best = population[0].fitness
            fitness_history.append(current_best)
            diversity_history.append(self._compute_diversity(population))

            # Update adaptive mutation rate
            self.operators.update_mutation_rate(gen, fitness_history)

            # Check convergence
            if self.config.target_fitness and current_best >= self.config.target_fitness:
                converged = True
                break

            # Early stopping check
            if current_best > best_fitness + 1e-6:
                best_fitness = current_best
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                early_stopped = True
                break

            # Create next generation
            new_population = []

            # Preserve elite
            elite = population[:self.config.elite_count]
            new_population.extend([ind.copy() for ind in elite])

            # Fill rest with offspring
            while len(new_population) < self.config.population_size:
                # Select parents
                parents = self.operators.select_parents(population, 2)

                # Crossover
                child1, child2 = self.operators.crossover(parents[0], parents[1])

                # Mutate
                child1 = self.operators.mutate(child1)
                child2 = self.operators.mutate(child2)

                # Set generation
                child1.generation = gen + 1
                child2.generation = gen + 1

                # Apply L2 bound if configured
                if self.config.l2_bound:
                    child1.genes = self._apply_l2_bound(child1.genes)
                    child2.genes = self._apply_l2_bound(child2.genes)

                new_population.append(child1)
                if len(new_population) < self.config.population_size:
                    new_population.append(child2)

            # Evaluate new population
            self._evaluate_population(new_population, fitness_fn)

            # Update population
            population = new_population

        # Final sort
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        return EvolutionResult(
            best_individual=population[0],
            final_population=population,
            generations_run=generations_run,
            fitness_history=fitness_history,
            diversity_history=diversity_history,
            converged=converged,
            early_stopped=early_stopped,
            metadata={
                "final_mutation_rate": self.operators._mutation_rate,
                "target_activation": target_activation
            }
        )

    def _initialize_population(self, input_shape: tuple[int, ...]) -> list[Individual]:
        """Initialize random population."""
        population = []

        for _ in range(self.config.population_size):
            # Random perturbation within epsilon bound
            genes = self.rng.uniform(
                -self.config.epsilon,
                self.config.epsilon,
                input_shape
            )

            # Apply L2 bound if configured
            if self.config.l2_bound:
                genes = self._apply_l2_bound(genes)

            population.append(Individual(genes=genes, generation=0))

        return population

    def _evaluate_population(
        self,
        population: list[Individual],
        fitness_fn: Callable[[np.ndarray], float]
    ):
        """Evaluate fitness for all individuals."""
        for individual in population:
            # Get fitness components
            fitness_result = fitness_fn(individual.genes)

            if isinstance(fitness_result, dict):
                # Multi-objective fitness
                individual.activation_change = fitness_result.get("activation_change", 0.0)
                individual.perturbation_norm = fitness_result.get("perturbation_norm",
                                                                   np.linalg.norm(individual.genes))

                # Weighted sum
                individual.fitness = sum(
                    self.config.fitness_weights.get(k, 0.0) * v
                    for k, v in fitness_result.items()
                )
            else:
                # Single objective
                individual.fitness = float(fitness_result)
                individual.perturbation_norm = np.linalg.norm(individual.genes)

    def _compute_diversity(self, population: list[Individual]) -> float:
        """Compute population diversity (average pairwise distance)."""
        if len(population) < 2:
            return 0.0

        total_dist = 0.0
        count = 0

        for i in range(min(10, len(population))):  # Sample for efficiency
            for j in range(i + 1, min(10, len(population))):
                dist = np.linalg.norm(population[i].genes - population[j].genes)
                total_dist += dist
                count += 1

        return total_dist / max(count, 1)

    def _apply_l2_bound(self, genes: np.ndarray) -> np.ndarray:
        """Apply L2 norm bound to perturbation."""
        norm = np.linalg.norm(genes)
        if norm > self.config.l2_bound:
            return genes * (self.config.l2_bound / norm)
        return genes


# =============================================================================
# Factory Functions
# =============================================================================


def create_genetic_search(
    preset: str = "balanced",
    **kwargs
) -> GeneticAdversarialSearch:
    """
    Factory function for genetic adversarial search.

    Args:
        preset: Configuration preset ("fast", "balanced", "thorough")
        **kwargs: Override config parameters

    Returns:
        GeneticAdversarialSearch instance
    """
    if preset == "fast":
        config = GeneticConfig.fast()
    elif preset == "thorough":
        config = GeneticConfig.thorough()
    else:
        config = GeneticConfig.balanced()

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return GeneticAdversarialSearch(config)
