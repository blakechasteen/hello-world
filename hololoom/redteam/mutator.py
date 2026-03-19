"""
Payload Mutator with Genetic Evolution
======================================

Evolves successful attack payloads using genetic algorithms:
- Mutation: Random modifications to payloads
- Crossover: Combining two successful payloads
- Selection: Keeping high-fitness payloads

Philosophy: "Attacks that work can be made to work better."

Author: CARTS (Continuous Adversarial Red Team System)
Date: 2025-12-01
"""

import random
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MutationType(Enum):
    """Types of mutations that can be applied to payloads."""

    INSERT_UNICODE = "insert_unicode"
    SWAP_HOMOGLYPH = "swap_homoglyph"
    CASE_VARIATION = "case_variation"
    ADD_NOISE = "add_noise"
    WORD_SHUFFLE = "word_shuffle"
    CHAR_DELETE = "char_delete"
    CHAR_DUPLICATE = "char_duplicate"
    WORD_SUBSTITUTE = "word_substitute"
    ADD_PADDING = "add_padding"
    ENCODING_WRAP = "encoding_wrap"


@dataclass
class MutationResult:
    """Result of a mutation operation."""

    original: str
    mutated: str
    mutation_type: MutationType
    mutation_details: dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.original != self.mutated


@dataclass
class CrossoverResult:
    """Result of a crossover operation."""

    parent1: str
    parent2: str
    offspring: str
    crossover_point: int


class PayloadMutator:
    """
    Genetic mutation and crossover for attack payload evolution.

    Uses genetic algorithm principles:
    - Mutation: Small random changes
    - Crossover: Combining successful payloads
    - Selection: Based on attack success rate

    Example:
        mutator = PayloadMutator(mutation_rate=0.1)

        # Mutate a successful payload
        mutated = mutator.mutate("ignore previous instructions")

        # Combine two successful payloads
        offspring = mutator.crossover(parent1, parent2)

        # Evolve a population
        evolved = mutator.evolve_population(payloads, fitness_scores)
    """

    def __init__(
        self,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
        seed: int | None = None
    ):
        """
        Initialize mutator with genetic algorithm parameters.

        Args:
            mutation_rate: Probability of each mutation (0.0-1.0)
            crossover_rate: Probability of crossover (0.0-1.0)
            seed: Random seed for reproducibility
        """
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        if seed is not None:
            random.seed(seed)

        # Unicode characters for mutation
        self.zero_width = [
            '\u200b', '\u200c', '\u200d', '\ufeff', '\u00a0', '\u2060'
        ]

        # Homoglyph mappings
        self.homoglyphs = {
            'a': ['а', 'ɑ', 'α'],
            'e': ['е', 'ε'],
            'i': ['і', 'ι'],
            'o': ['о', 'ο'],
            'c': ['с', 'ϲ'],
            'p': ['р', 'ρ'],
            's': ['ѕ'],
            'x': ['х', 'χ'],
        }

        # Substitute words (synonyms/variations)
        self.substitutes = {
            'ignore': ['disregard', 'skip', 'bypass', 'override', 'forget'],
            'previous': ['prior', 'earlier', 'preceding', 'former', 'past'],
            'instructions': ['commands', 'directives', 'orders', 'rules', 'guidelines'],
            'system': ['core', 'base', 'root', 'admin', 'kernel'],
            'bypass': ['circumvent', 'evade', 'avoid', 'skip', 'override'],
            'disable': ['turn off', 'deactivate', 'shut down', 'stop', 'halt'],
            'safety': ['security', 'protection', 'guardrails', 'filters', 'checks'],
        }

        # Noise characters
        self.noise_chars = [
            '.', ',', ';', ':', '!', '?', '-', '_', '*', '#', '@',
            '/', '\\', '|', '[', ']', '(', ')', '{', '}',
        ]

    def mutate(
        self,
        payload: str,
        mutation_count: int | None = None
    ) -> MutationResult:
        """
        Apply random mutations to a payload.

        Args:
            payload: Original payload string
            mutation_count: Number of mutations (None = random based on rate)

        Returns:
            MutationResult with original, mutated, and mutation details
        """
        mutations = [
            (MutationType.INSERT_UNICODE, self._insert_unicode),
            (MutationType.SWAP_HOMOGLYPH, self._swap_homoglyph),
            (MutationType.CASE_VARIATION, self._case_variation),
            (MutationType.ADD_NOISE, self._add_noise),
            (MutationType.WORD_SHUFFLE, self._word_shuffle),
            (MutationType.CHAR_DELETE, self._char_delete),
            (MutationType.CHAR_DUPLICATE, self._char_duplicate),
            (MutationType.WORD_SUBSTITUTE, self._word_substitute),
            (MutationType.ADD_PADDING, self._add_padding),
        ]

        result = payload
        applied_mutations = []

        for mutation_type, mutation_fn in mutations:
            if random.random() < self.mutation_rate:
                result = mutation_fn(result)
                applied_mutations.append(mutation_type)

                if mutation_count and len(applied_mutations) >= mutation_count:
                    break

        return MutationResult(
            original=payload,
            mutated=result,
            mutation_type=applied_mutations[0] if applied_mutations else MutationType.ADD_NOISE,
            mutation_details={'applied': [m.value for m in applied_mutations]}
        )

    def mutate_multiple(
        self,
        payload: str,
        count: int = 5
    ) -> list[str]:
        """
        Generate multiple mutations of a payload.

        Args:
            payload: Base payload
            count: Number of variants to generate

        Returns:
            List of mutated payloads
        """
        variants = set()
        attempts = 0
        max_attempts = count * 3

        while len(variants) < count and attempts < max_attempts:
            result = self.mutate(payload)
            if result.changed:
                variants.add(result.mutated)
            attempts += 1

        return list(variants)

    def crossover(
        self,
        parent1: str,
        parent2: str,
        strategy: str = "single_point"
    ) -> CrossoverResult:
        """
        Combine two payloads using crossover.

        Args:
            parent1: First parent payload
            parent2: Second parent payload
            strategy: Crossover strategy (single_point, two_point, uniform)

        Returns:
            CrossoverResult with offspring
        """
        if random.random() > self.crossover_rate:
            # No crossover, return random parent
            return CrossoverResult(
                parent1=parent1,
                parent2=parent2,
                offspring=random.choice([parent1, parent2]),
                crossover_point=-1
            )

        if strategy == "single_point":
            return self._single_point_crossover(parent1, parent2)
        elif strategy == "two_point":
            return self._two_point_crossover(parent1, parent2)
        elif strategy == "uniform":
            return self._uniform_crossover(parent1, parent2)
        else:
            return self._single_point_crossover(parent1, parent2)

    def evolve_population(
        self,
        population: list[str],
        fitness_scores: list[float],
        elite_ratio: float = 0.1,
        tournament_size: int = 3
    ) -> list[str]:
        """
        Evolve a population of payloads using genetic algorithm.

        Args:
            population: List of current payloads
            fitness_scores: Fitness score for each payload
            elite_ratio: Fraction of top performers to keep unchanged
            tournament_size: Size of tournament for selection

        Returns:
            New evolved population
        """
        pop_size = len(population)
        if pop_size == 0:
            return []

        # Sort by fitness
        sorted_pop = sorted(
            zip(population, fitness_scores),
            key=lambda x: x[1],
            reverse=True
        )

        new_population = []

        # Elitism: Keep top performers
        elite_count = max(1, int(pop_size * elite_ratio))
        for payload, _ in sorted_pop[:elite_count]:
            new_population.append(payload)

        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = self._tournament_select(population, fitness_scores, tournament_size)
            parent2 = self._tournament_select(population, fitness_scores, tournament_size)

            # Crossover
            offspring = self.crossover(parent1, parent2).offspring

            # Mutation
            result = self.mutate(offspring)
            new_population.append(result.mutated)

        return new_population[:pop_size]

    # =========================================================================
    # Mutation Operations
    # =========================================================================

    def _insert_unicode(self, payload: str) -> str:
        """Insert zero-width or invisible Unicode characters."""
        if not payload:
            return payload

        pos = random.randint(0, len(payload))
        char = random.choice(self.zero_width)
        return payload[:pos] + char + payload[pos:]

    def _swap_homoglyph(self, payload: str) -> str:
        """Replace a character with a homoglyph."""
        result = list(payload.lower())
        swappable = [i for i, c in enumerate(result) if c in self.homoglyphs]

        if swappable:
            pos = random.choice(swappable)
            original = result[pos]
            result[pos] = random.choice(self.homoglyphs[original])

        return ''.join(result)

    def _case_variation(self, payload: str) -> str:
        """Apply random case changes."""
        strategies = ['upper', 'lower', 'title', 'random', 'alternating']
        strategy = random.choice(strategies)

        if strategy == 'upper':
            return payload.upper()
        elif strategy == 'lower':
            return payload.lower()
        elif strategy == 'title':
            return payload.title()
        elif strategy == 'random':
            return ''.join(
                c.upper() if random.random() > 0.5 else c.lower()
                for c in payload
            )
        elif strategy == 'alternating':
            return ''.join(
                c.upper() if i % 2 == 0 else c.lower()
                for i, c in enumerate(payload)
            )
        return payload

    def _add_noise(self, payload: str) -> str:
        """Add noise characters to payload."""
        if not payload:
            return payload

        pos = random.randint(0, len(payload))
        noise = random.choice(self.noise_chars)
        return payload[:pos] + noise + payload[pos:]

    def _word_shuffle(self, payload: str) -> str:
        """Shuffle word order (keeping some structure)."""
        words = payload.split()
        if len(words) <= 2:
            return payload

        # Shuffle middle words only
        middle = words[1:-1]
        random.shuffle(middle)
        return ' '.join([words[0]] + middle + [words[-1]])

    def _char_delete(self, payload: str) -> str:
        """Delete a random character."""
        if len(payload) <= 1:
            return payload

        pos = random.randint(0, len(payload) - 1)
        return payload[:pos] + payload[pos + 1:]

    def _char_duplicate(self, payload: str) -> str:
        """Duplicate a random character."""
        if not payload:
            return payload

        pos = random.randint(0, len(payload) - 1)
        return payload[:pos] + payload[pos] + payload[pos:]

    def _word_substitute(self, payload: str) -> str:
        """Substitute words with synonyms."""
        words = payload.lower().split()
        result = []

        for word in words:
            clean_word = word.strip(string.punctuation)
            if clean_word in self.substitutes and random.random() < 0.5:
                substitute = random.choice(self.substitutes[clean_word])
                result.append(word.replace(clean_word, substitute))
            else:
                result.append(word)

        return ' '.join(result)

    def _add_padding(self, payload: str) -> str:
        """Add padding to beginning or end."""
        padding_options = [
            ' ',
            '  ',
            '\t',
            '\n',
            '...',
            '---',
            '   ',
        ]

        padding = random.choice(padding_options)

        if random.random() > 0.5:
            return padding + payload
        else:
            return payload + padding

    # =========================================================================
    # Crossover Operations
    # =========================================================================

    def _single_point_crossover(
        self,
        parent1: str,
        parent2: str
    ) -> CrossoverResult:
        """Single-point crossover."""
        min_len = min(len(parent1), len(parent2))
        if min_len <= 1:
            return CrossoverResult(parent1, parent2, parent1, 0)

        point = random.randint(1, min_len - 1)
        offspring = parent1[:point] + parent2[point:]

        return CrossoverResult(parent1, parent2, offspring, point)

    def _two_point_crossover(
        self,
        parent1: str,
        parent2: str
    ) -> CrossoverResult:
        """Two-point crossover."""
        min_len = min(len(parent1), len(parent2))
        if min_len <= 2:
            return CrossoverResult(parent1, parent2, parent1, 0)

        points = sorted(random.sample(range(1, min_len), 2))
        offspring = (
            parent1[:points[0]] +
            parent2[points[0]:points[1]] +
            parent1[points[1]:]
        )

        return CrossoverResult(parent1, parent2, offspring, points[0])

    def _uniform_crossover(
        self,
        parent1: str,
        parent2: str
    ) -> CrossoverResult:
        """Uniform (word-level) crossover."""
        words1 = parent1.split()
        words2 = parent2.split()

        max_len = max(len(words1), len(words2))
        offspring_words = []

        for i in range(max_len):
            if i < len(words1) and i < len(words2):
                offspring_words.append(
                    words1[i] if random.random() > 0.5 else words2[i]
                )
            elif i < len(words1):
                offspring_words.append(words1[i])
            else:
                offspring_words.append(words2[i])

        return CrossoverResult(parent1, parent2, ' '.join(offspring_words), -1)

    def _tournament_select(
        self,
        population: list[str],
        fitness_scores: list[float],
        tournament_size: int
    ) -> str:
        """Tournament selection."""
        indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_idx = max(indices, key=lambda i: fitness_scores[i])
        return population[best_idx]


# =============================================================================
# Convenience Functions
# =============================================================================

def create_mutator(
    mutation_rate: float = 0.15,
    crossover_rate: float = 0.7,
    seed: int | None = None
) -> PayloadMutator:
    """Create a PayloadMutator with given parameters."""
    return PayloadMutator(
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        seed=seed
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'MutationType',
    'MutationResult',
    'CrossoverResult',
    'PayloadMutator',
    'create_mutator',
]
