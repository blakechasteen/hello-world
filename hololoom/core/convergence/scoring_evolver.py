"""
Scoring Term Evolver — AlphaEvolve Pattern for Self-Improving Decisions
=======================================================================

Wire 12 of the structured AI integration roadmap.

Applies the AlphaEvolve pattern to HoloLoom's additive scoring system
(Design Principle #4). The system evolves its own scoring term weights
and discovers new scoring configurations.

Architecture (mirrors AlphaEvolve's 5 components):
1. **Population**: MAP-Elites grid of scoring configurations
   - Each cell holds the best config for a (quality, diversity) bin
   - Maintains both quality AND diversity simultaneously
2. **Mutator**: Generates new configs by perturbing weights or toggling terms
3. **Evaluator**: Scores a config by running it against historical outcomes
4. **Selector**: Thompson Sampling picks which configs to mutate next
5. **Controller**: Manages the evolution loop

Based on: AlphaEvolve (DeepMind, May 2025) — found 48-multiplication
algorithm for 4x4 complex matrix multiply, beating Strassen's 1969 result.
OpenEvolve (Apache 2.0) provides the pip-installable open-source variant.

Usage:
    evolver = ScoringEvolver(
        term_names=["w_tc", "w_urgency", "w_variety", "w_craving"],
        initial_weights={"w_tc": 25, "w_urgency": 15, "w_variety": 5, "w_craving": 6},
    )
    evolver.seed_population(n=10)

    # Evolution loop
    for _ in range(100):
        config = evolver.select_parent()
        child = evolver.mutate(config)
        fitness = evolver.evaluate(child, historical_outcomes)
        evolver.record(child, fitness)

    best = evolver.best_config()

Author: Claude Code (Structured AI Wiring - Wire 12)
Date: 2026-03-18
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class ScoringConfig:
    """A candidate scoring configuration (individual in the population)."""
    weights: dict[str, float]       # {term_name: weight}
    fitness: float = 0.0            # Evaluated fitness score
    generation: int = 0             # Which generation it was created in
    parent_id: int | None = None # Which config it was mutated from
    config_id: int = 0              # Unique identifier

    def score(self, term_values: dict[str, float]) -> float:
        """Compute additive score using this config's weights."""
        return sum(
            self.weights.get(term, 0.0) * term_values.get(term, 0.0)
            for term in self.weights
        )


@dataclass
class EvolutionStats:
    """Statistics from the evolution process."""
    generations: int
    population_size: int
    best_fitness: float
    mean_fitness: float
    fitness_trajectory: list[float]  # Best fitness per generation
    diversity: float                 # Fraction of unique configs in population


# ============================================================================
# MAP-Elites Population
# ============================================================================

class _MAPElitesGrid:
    """
    MAP-Elites population: maintains diversity by binning configs
    into a grid of (quality, diversity) cells. Each cell keeps only
    the highest-fitness config.
    """

    def __init__(self, bins: int = 10):
        self.bins = bins
        self.grid: dict[tuple[int, int], ScoringConfig] = {}
        self._all_configs: list[ScoringConfig] = []

    def _compute_features(self, config: ScoringConfig) -> tuple[int, int]:
        """Map config to a 2D feature cell: (weight_spread, weight_sum)."""
        weights = list(config.weights.values())
        if not weights:
            return (0, 0)

        # Feature 1: weight spread (diversity of weights)
        spread = float(np.std(weights)) / (float(np.mean(np.abs(weights))) + 1e-9)
        spread_bin = min(int(spread * self.bins), self.bins - 1)

        # Feature 2: total weight magnitude
        total = sum(abs(w) for w in weights)
        max_expected = 100.0 * len(weights)
        total_bin = min(int((total / max_expected) * self.bins), self.bins - 1)

        return (spread_bin, total_bin)

    def add(self, config: ScoringConfig) -> bool:
        """
        Add config to population. Returns True if it was placed (new or better).
        """
        cell = self._compute_features(config)
        existing = self.grid.get(cell)

        if existing is None or config.fitness > existing.fitness:
            self.grid[cell] = config
            self._all_configs.append(config)
            return True
        return False

    def sample(self) -> ScoringConfig | None:
        """Sample a config from the population (uniform over occupied cells)."""
        if not self.grid:
            return None
        cells = list(self.grid.values())
        return cells[np.random.randint(len(cells))]

    def best(self) -> ScoringConfig | None:
        """Return the highest-fitness config in the population."""
        if not self.grid:
            return None
        return max(self.grid.values(), key=lambda c: c.fitness)

    def all_configs(self) -> list[ScoringConfig]:
        """Return all configs currently in the grid."""
        return list(self.grid.values())

    def size(self) -> int:
        return len(self.grid)

    def diversity(self) -> float:
        """Fraction of grid cells that are occupied."""
        return len(self.grid) / max(self.bins * self.bins, 1)


# ============================================================================
# Scoring Evolver
# ============================================================================

class ScoringEvolver:
    """
    Evolves additive scoring configurations using the AlphaEvolve pattern.

    The evolver maintains a MAP-Elites population of scoring configs,
    uses Thompson Sampling to select parents for mutation, and evaluates
    fitness against historical outcomes.

    This is the self-improving decision system: the system literally
    evolves its own scoring function weights.
    """

    def __init__(
        self,
        term_names: list[str],
        initial_weights: dict[str, float] | None = None,
        mutation_rate: float = 0.3,
        mutation_scale: float = 5.0,
        bins: int = 10,
    ):
        """
        Args:
            term_names: Names of scoring terms (e.g., ["w_tc", "w_urgency", ...])
            initial_weights: Starting weights (default: uniform 10.0)
            mutation_rate: Probability of mutating each weight (0-1)
            mutation_scale: Standard deviation of weight perturbation
            bins: MAP-Elites grid resolution per dimension
        """
        self.term_names = list(term_names)
        self.initial_weights = initial_weights or dict.fromkeys(term_names, 10.0)
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale

        self._population = _MAPElitesGrid(bins=bins)
        self._generation = 0
        self._next_id = 0
        self._fitness_history: list[float] = []

        # Thompson Sampling: per-config prior on "is this a good parent?"
        self._parent_priors: dict[int, list[float]] = {}  # {config_id: [α, β]}

        logger.info(
            f"ScoringEvolver initialized "
            f"(terms={len(term_names)}, mutation_rate={mutation_rate}, "
            f"mutation_scale={mutation_scale})"
        )

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def seed_population(self, n: int = 10) -> list[ScoringConfig]:
        """
        Seed the population with random perturbations of initial weights.

        Args:
            n: Number of initial configs

        Returns:
            List of seeded configs
        """
        seeded = []

        # First config: exact initial weights
        base = ScoringConfig(
            weights=dict(self.initial_weights),
            config_id=self._new_id(),
            generation=0,
        )
        self._population.add(base)
        seeded.append(base)

        # Remaining: random perturbations
        for _ in range(n - 1):
            weights = {}
            for term in self.term_names:
                base_w = self.initial_weights.get(term, 10.0)
                perturbation = np.random.normal(0, self.mutation_scale)
                weights[term] = max(0.0, base_w + perturbation)

            config = ScoringConfig(
                weights=weights,
                config_id=self._new_id(),
                generation=0,
            )
            self._population.add(config)
            seeded.append(config)

        logger.info(f"Seeded population with {len(seeded)} configs")
        return seeded

    def select_parent(self) -> ScoringConfig:
        """
        Select a parent config for mutation using Thompson Sampling.

        Configs that have produced high-fitness children get higher
        selection probability over time.
        """
        candidates = self._population.all_configs()
        if not candidates:
            # No population yet — return initial
            return ScoringConfig(
                weights=dict(self.initial_weights),
                config_id=self._new_id(),
            )

        # Thompson sample: score each candidate
        scores = []
        for config in candidates:
            alpha, beta = self._parent_priors.get(config.config_id, [1.0, 1.0])
            ts_score = float(np.random.beta(alpha, beta))
            scores.append(ts_score)

        best_idx = int(np.argmax(scores))
        return candidates[best_idx]

    def mutate(self, parent: ScoringConfig) -> ScoringConfig:
        """
        Create a child config by mutating a parent.

        Each weight has mutation_rate probability of being perturbed
        by a Gaussian with scale mutation_scale.
        """
        self._generation += 1
        child_weights = {}

        for term in self.term_names:
            parent_w = parent.weights.get(term, 10.0)

            if np.random.random() < self.mutation_rate:
                # Mutate: Gaussian perturbation
                perturbation = np.random.normal(0, self.mutation_scale)
                child_weights[term] = max(0.0, parent_w + perturbation)
            else:
                # Keep parent's weight
                child_weights[term] = parent_w

        child = ScoringConfig(
            weights=child_weights,
            config_id=self._new_id(),
            generation=self._generation,
            parent_id=parent.config_id,
        )
        return child

    def evaluate(
        self,
        config: ScoringConfig,
        historical_data: list[dict[str, float]],
        ground_truth: list[int],
    ) -> float:
        """
        Evaluate a config against historical outcomes.

        For each historical example, computes scores for all options
        using the config's weights, then checks if the highest-scored
        option matches ground truth.

        Args:
            config: Scoring config to evaluate
            historical_data: List of dicts with term values per option.
                           Each dict has keys like "option_0_w_tc", "option_1_w_tc", etc.
            ground_truth: List of correct option indices

        Returns:
            Fitness score (accuracy: fraction of correct top-1 predictions)
        """
        if not historical_data or not ground_truth:
            return 0.5  # No data — return neutral fitness

        correct = 0
        total = min(len(historical_data), len(ground_truth))

        for i in range(total):
            example = historical_data[i]
            true_choice = ground_truth[i]

            # Find all options in this example
            options = set()
            for key in example:
                parts = key.split("_", 2)
                if len(parts) >= 2 and parts[0] == "option":
                    try:
                        options.add(int(parts[1]))
                    except ValueError:
                        pass

            if not options:
                continue

            # Score each option
            best_option = -1
            best_score = -float('inf')

            for opt_idx in sorted(options):
                term_values = {}
                for term in self.term_names:
                    key = f"option_{opt_idx}_{term}"
                    term_values[term] = example.get(key, 0.0)

                score = config.score(term_values)
                if score > best_score:
                    best_score = score
                    best_option = opt_idx

            if best_option == true_choice:
                correct += 1

        fitness = correct / max(total, 1)
        config.fitness = fitness
        return fitness

    def record(self, config: ScoringConfig, fitness: float) -> bool:
        """
        Record a config and its fitness. Returns True if it entered the population.

        Also updates Thompson Sampling prior for the parent.
        """
        config.fitness = fitness

        # Update parent's Thompson prior
        if config.parent_id is not None:
            if config.parent_id not in self._parent_priors:
                self._parent_priors[config.parent_id] = [1.0, 1.0]

            if fitness > 0.5:
                self._parent_priors[config.parent_id][0] += fitness
            else:
                self._parent_priors[config.parent_id][1] += (1.0 - fitness)

        # Track best fitness
        self._fitness_history.append(fitness)

        # Try to add to population
        placed = self._population.add(config)
        if placed:
            logger.debug(
                f"Gen {config.generation}: config {config.config_id} placed "
                f"(fitness={fitness:.3f})"
            )
        return placed

    def best_config(self) -> ScoringConfig | None:
        """Return the highest-fitness config in the population."""
        return self._population.best()

    def stats(self) -> EvolutionStats:
        """Return evolution statistics."""
        configs = self._population.all_configs()
        fitnesses = [c.fitness for c in configs] if configs else [0.0]

        return EvolutionStats(
            generations=self._generation,
            population_size=self._population.size(),
            best_fitness=max(fitnesses),
            mean_fitness=float(np.mean(fitnesses)),
            fitness_trajectory=list(self._fitness_history),
            diversity=self._population.diversity(),
        )


# ============================================================================
# Factory
# ============================================================================

def create_scoring_evolver(
    term_names: list[str],
    initial_weights: dict[str, float] | None = None,
    mutation_rate: float = 0.3,
    mutation_scale: float = 5.0,
) -> ScoringEvolver:
    """Create a scoring term evolver."""
    return ScoringEvolver(
        term_names=term_names,
        initial_weights=initial_weights,
        mutation_rate=mutation_rate,
        mutation_scale=mutation_scale,
    )
