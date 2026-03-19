"""
Cellular Automata — SA3.1-3.3
==============================

Discrete dynamical systems with local update rules. From Wolfram's
elementary CA (1D, 2-state, radius-1) to Conway's Game of Life (2D),
with Langton's lambda parameter for edge-of-chaos classification.

Classes:
    ElementaryCA: Wolfram's 256 1D cellular automata
    GameOfLife: Conway's 2D cellular automaton
    LangtonParameter: Edge-of-chaos measure for rule tables

Key Concepts:
    - Local rules: next state depends only on neighborhood
    - Wolfram classes: I (fixed), II (periodic), III (chaotic), IV (complex)
    - Lambda parameter: fraction of non-quiescent transitions
    - Edge of chaos: complex behavior at critical lambda ~ 0.27

Applications:
    - Pattern emergence in HoloLoom memory dynamics
    - Self-organizing feature representations
    - Complexity measures for reasoning traces

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ElementaryCA:
    """
    Wolfram's elementary cellular automata.

    256 possible rules for 1D, 2-state, radius-1 automata.
    Rule number encodes the output for each of the 8 possible
    neighborhood configurations (left, center, right).

    Rule 30: chaotic (used by Mathematica for random numbers)
    Rule 110: Turing-complete (proven by Cook, 2004)
    Rule 90: Sierpinski triangle (XOR of neighbors)
    """

    @staticmethod
    def evolve(
        rule: int,
        initial: np.ndarray,
        n_steps: int,
        boundary: str = "periodic",
    ) -> np.ndarray:
        """
        Evolve a 1D elementary CA.

        Args:
            rule: Rule number 0-255
            initial: Initial configuration, shape (width,) of 0s and 1s
            n_steps: Number of time steps to evolve
            boundary: Boundary condition: "periodic" (wrap) or "fixed" (0-padded)

        Returns:
            Space-time diagram, shape (n_steps + 1, width)
        """
        if not 0 <= rule <= 255:
            raise ValueError(f"Rule must be 0-255, got {rule}")

        width = len(initial)
        grid = np.zeros((n_steps + 1, width), dtype=int)
        grid[0] = initial.copy()

        # Decode rule into lookup table
        lookup = np.array([(rule >> i) & 1 for i in range(8)], dtype=int)

        for t in range(n_steps):
            row = grid[t]
            new_row = np.zeros(width, dtype=int)

            for i in range(width):
                if boundary == "periodic":
                    left = row[(i - 1) % width]
                    right = row[(i + 1) % width]
                else:
                    left = row[i - 1] if i > 0 else 0
                    right = row[i + 1] if i < width - 1 else 0

                center = row[i]
                neighborhood = (left << 2) | (center << 1) | right
                new_row[i] = lookup[neighborhood]

            grid[t + 1] = new_row

        return grid

    @staticmethod
    def single_cell_initial(width: int) -> np.ndarray:
        """Standard initial condition: single 1 in the center."""
        initial = np.zeros(width, dtype=int)
        initial[width // 2] = 1
        return initial

    @staticmethod
    def random_initial(width: int, density: float = 0.5) -> np.ndarray:
        """Random initial condition with given density of 1s."""
        return (np.random.random(width) < density).astype(int)

    @staticmethod
    def rule_table(rule: int) -> dict[tuple[int, int, int], int]:
        """
        Decode rule number into explicit lookup table.

        Returns:
            Dict mapping (left, center, right) -> next_state
        """
        table = {}
        for i in range(8):
            left = (i >> 2) & 1
            center = (i >> 1) & 1
            right = i & 1
            table[(left, center, right)] = (rule >> i) & 1
        return table

    @staticmethod
    def entropy_density(spacetime: np.ndarray) -> float:
        """
        Compute spatial entropy density of the final configuration.

        Higher entropy = more disordered (Class III).
        Lower entropy = more ordered (Class I/II).
        """
        final = spacetime[-1]
        p1 = final.mean()
        p0 = 1 - p1
        if p0 < 1e-10 or p1 < 1e-10:
            return 0.0
        return -(p0 * np.log2(p0) + p1 * np.log2(p1))

    @staticmethod
    def classify_rule(rule: int, width: int = 201, steps: int = 200) -> str:
        """
        Classify a rule into Wolfram's four classes.

        Uses heuristics based on final state properties:
        - Class I: homogeneous (all 0 or all 1)
        - Class II: periodic (low unique patterns)
        - Class III: chaotic (high entropy, no clear structure)
        - Class IV: complex (moderate entropy, long transients)

        Args:
            rule: Rule number 0-255
            width: Grid width for testing
            steps: Number of evolution steps

        Returns:
            Classification string: "I", "II", "III", or "IV"
        """
        initial = ElementaryCA.single_cell_initial(width)
        grid = ElementaryCA.evolve(rule, initial, steps)

        final = grid[-1]
        density = final.mean()

        # Check for homogeneity (Class I)
        if density < 0.01 or density > 0.99:
            return "I"

        # Check for periodicity (Class II)
        # Compare last few rows for repeats
        period_found = False
        for p in range(1, min(20, steps)):
            if np.array_equal(grid[-1], grid[-(p + 1)]):
                period_found = True
                break

        if period_found:
            return "II"

        # Entropy-based distinction between III and IV
        entropy = ElementaryCA.entropy_density(grid)

        # Check for localized structures (Class IV indicator)
        # Compute column-wise variance as a proxy
        col_variance = grid[steps // 2:].var(axis=0)
        variance_of_variance = col_variance.var()

        if entropy > 0.9 and variance_of_variance < 0.01:
            return "III"  # Uniform chaos

        return "IV"  # Complex/edge of chaos


class GameOfLife:
    """
    Conway's Game of Life — 2D cellular automaton.

    Rules (B3/S23):
        - Birth: dead cell with exactly 3 live neighbors becomes alive
        - Survival: live cell with 2 or 3 live neighbors survives
        - Death: all other live cells die (underpopulation or overcrowding)

    Supports toroidal (periodic) boundary conditions.
    """

    @staticmethod
    def evolve(
        initial: np.ndarray,
        n_steps: int,
        boundary: str = "periodic",
    ) -> np.ndarray:
        """
        Evolve the Game of Life.

        Args:
            initial: Initial grid, shape (height, width) of 0s and 1s
            n_steps: Number of generations to evolve
            boundary: "periodic" (torus) or "fixed" (dead border)

        Returns:
            Space-time evolution, shape (n_steps + 1, height, width)
        """
        h, w = initial.shape
        history = np.zeros((n_steps + 1, h, w), dtype=int)
        history[0] = initial.copy()

        for t in range(n_steps):
            grid = history[t]
            neighbors = GameOfLife._count_neighbors(grid, boundary)
            new_grid = np.zeros_like(grid)

            # Birth: dead cell with exactly 3 neighbors
            new_grid[(grid == 0) & (neighbors == 3)] = 1
            # Survival: live cell with 2 or 3 neighbors
            new_grid[(grid == 1) & ((neighbors == 2) | (neighbors == 3))] = 1

            history[t + 1] = new_grid

        return history

    @staticmethod
    def _count_neighbors(grid: np.ndarray, boundary: str) -> np.ndarray:
        """Count live neighbors for each cell."""
        h, w = grid.shape
        counts = np.zeros_like(grid)

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                if boundary == "periodic":
                    shifted = np.roll(np.roll(grid, di, axis=0), dj, axis=1)
                else:
                    shifted = np.zeros_like(grid)
                    si = max(0, di)
                    ei = h + min(0, di)
                    sj = max(0, dj)
                    ej = w + min(0, dj)
                    src_si = max(0, -di)
                    src_ei = h - max(0, di)
                    src_sj = max(0, -dj)
                    src_ej = w - max(0, dj)
                    shifted[si:ei, sj:ej] = grid[src_si:src_ei, src_sj:src_ej]
                counts += shifted

        return counts

    @staticmethod
    def random_grid(height: int, width: int, density: float = 0.3) -> np.ndarray:
        """Random initial grid."""
        return (np.random.random((height, width)) < density).astype(int)

    @staticmethod
    def glider(height: int = 20, width: int = 20, pos: tuple[int, int] = (1, 1)) -> np.ndarray:
        """Standard glider pattern."""
        grid = np.zeros((height, width), dtype=int)
        r, c = pos
        # Glider: period 4, moves diagonally
        pattern = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        for dr, dc in pattern:
            grid[(r + dr) % height, (c + dc) % width] = 1
        return grid

    @staticmethod
    def population(history: np.ndarray) -> np.ndarray:
        """Population count over time."""
        return history.sum(axis=(1, 2))

    @staticmethod
    def is_still_life(grid: np.ndarray, boundary: str = "periodic") -> bool:
        """Check if configuration is a still life (fixed point)."""
        evolved = GameOfLife.evolve(grid, 1, boundary)
        return np.array_equal(grid, evolved[1])

    @staticmethod
    def period(history: np.ndarray, max_check: int = 100) -> int:
        """
        Detect the period of an evolved CA.

        Args:
            history: Full space-time evolution
            max_check: Maximum period to check

        Returns:
            Period (1 = still life, 0 = no period found)
        """
        final = history[-1]
        for p in range(1, min(max_check, len(history))):
            if np.array_equal(final, history[-(p + 1)]):
                return p
        return 0


class LangtonParameter:
    """
    Langton's lambda parameter for cellular automata rule tables.

    Lambda measures the fraction of non-quiescent transitions:
        lambda = (K^N - n_q) / K^N

    where K = number of states, N = neighborhood size,
    n_q = number of transitions to the quiescent state.

    Critical lambda values:
        lambda ~ 0:   Class I  (all quiescent)
        lambda ~ 0.2: Class II (periodic)
        lambda ~ 0.27: Class IV (edge of chaos)
        lambda ~ 0.5: Class III (chaotic)
    """

    @staticmethod
    def compute(rule_table: dict, n_states: int = 2, quiescent: int = 0) -> float:
        """
        Compute lambda for a rule table.

        Args:
            rule_table: Maps neighborhood configuration -> output state
            n_states: Number of possible cell states
            quiescent: The quiescent/default state

        Returns:
            Lambda parameter in [0, 1]
        """
        total = len(rule_table)
        if total == 0:
            return 0.0

        n_quiescent = sum(1 for v in rule_table.values() if v == quiescent)
        return (total - n_quiescent) / total

    @staticmethod
    def compute_elementary(rule: int) -> float:
        """Compute lambda for a Wolfram elementary CA rule."""
        table = ElementaryCA.rule_table(rule)
        return LangtonParameter.compute(table, n_states=2, quiescent=0)

    @staticmethod
    def scan_rules(n_rules: int = 256) -> np.ndarray:
        """
        Compute lambda for all elementary CA rules.

        Returns:
            Array of lambda values, shape (n_rules,)
        """
        lambdas = np.zeros(n_rules)
        for rule in range(n_rules):
            lambdas[rule] = LangtonParameter.compute_elementary(rule)
        return lambdas

    @staticmethod
    def generate_rule_at_lambda(target_lambda: float, n_states: int = 2) -> dict:
        """
        Generate a random rule table with approximately the given lambda.

        Args:
            target_lambda: Desired lambda parameter
            n_states: Number of states

        Returns:
            Rule table (Dict mapping neighborhood -> output state)
        """
        n_configs = n_states ** 3  # radius-1, 1D
        n_non_quiescent = int(round(target_lambda * n_configs))

        # Start with all quiescent
        table = {}
        configs = []
        for left in range(n_states):
            for center in range(n_states):
                for right in range(n_states):
                    configs.append((left, center, right))
                    table[(left, center, right)] = 0

        # Randomly assign non-quiescent transitions
        indices = np.random.choice(len(configs), size=n_non_quiescent, replace=False)
        for idx in indices:
            config = configs[idx]
            table[config] = np.random.randint(1, n_states)

        return table

    @staticmethod
    def entropy_vs_lambda(
        n_samples: int = 50,
        width: int = 101,
        steps: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Measure entropy as a function of lambda across random rules.

        Demonstrates the phase transition at the edge of chaos.

        Args:
            n_samples: Number of lambda values to sample
            width: CA grid width
            steps: Evolution steps per trial

        Returns:
            (lambdas, entropies) arrays
        """
        lambdas = np.linspace(0, 1, n_samples)
        entropies = np.zeros(n_samples)

        for i, lam in enumerate(lambdas):
            # Find closest elementary rule
            best_rule = 0
            best_diff = 1.0
            for rule in range(256):
                diff = abs(LangtonParameter.compute_elementary(rule) - lam)
                if diff < best_diff:
                    best_diff = diff
                    best_rule = rule

            initial = ElementaryCA.random_initial(width)
            grid = ElementaryCA.evolve(best_rule, initial, steps)
            entropies[i] = ElementaryCA.entropy_density(grid)

        return lambdas, entropies
