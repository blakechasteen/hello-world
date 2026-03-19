"""
Approximation Algorithms for HoloLoom Warp Drive
==================================================

Randomized rounding, FPTAS for knapsack, Count-Min sketch,
and MinHash for Jaccard similarity estimation.

These are the algorithmic workhorses for problems where exact solutions
are NP-hard but approximate solutions suffice.

Author: HoloLoom Team
Date: 2026-03-19
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# RANDOMIZED ROUNDING
# ============================================================================

class RandomizedRounding:
    """
    Randomized rounding: convert fractional LP solution to integer solution.

    Given LP relaxation solution x* ∈ [0,1]ⁿ, round each variable
    independently: xᵢ = 1 with probability x*ᵢ.

    Expected objective = LP objective (unbiased).
    Constraints may be violated — use multiple trials and pick best feasible.

    For set cover: E[cost] ≤ OPT_LP ≤ OPT_IP. Repeat O(log n) times
    for high probability.
    """

    @staticmethod
    def round(lp_solution: np.ndarray, n_trials: int = 100,
              rng: np.random.Generator = None) -> np.ndarray:
        """
        Randomized rounding of a fractional solution.

        Args:
            lp_solution: Fractional solution x* ∈ [0,1]ⁿ.
            n_trials: Number of independent roundings to try.
            rng: Random number generator.

        Returns:
            Best integer solution found (binary 0/1 vector).
        """
        if rng is None:
            rng = np.random.default_rng(42)

        n = len(lp_solution)
        probs = np.clip(lp_solution, 0, 1)

        best_solution = None
        best_objective = -np.inf

        for _ in range(n_trials):
            # Round independently
            rounded = (rng.random(n) < probs).astype(float)

            # Objective: sum of selected items (maximize)
            obj = np.sum(rounded)

            if obj > best_objective:
                best_objective = obj
                best_solution = rounded.copy()

        logger.info(
            f"Randomized rounding: {n_trials} trials, "
            f"best objective={best_objective:.2f}"
        )
        return best_solution

    @staticmethod
    def round_with_constraints(
        lp_solution: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        n_trials: int = 100,
        rng: np.random.Generator = None
    ) -> tuple[np.ndarray, float]:
        """
        Randomized rounding with feasibility check.

        maximize c·x subject to Ax ≤ b, x ∈ {0,1}ⁿ.

        Returns best feasible solution and its objective value.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        n = len(lp_solution)
        probs = np.clip(lp_solution, 0, 1)

        best_solution = None
        best_objective = -np.inf

        for _ in range(n_trials):
            rounded = (rng.random(n) < probs).astype(float)

            # Check feasibility: Ax ≤ b
            if np.all(A @ rounded <= b + 1e-10):
                obj = c @ rounded
                if obj > best_objective:
                    best_objective = obj
                    best_solution = rounded.copy()

        if best_solution is None:
            logger.warning("No feasible solution found")
            return np.zeros(n), 0.0

        return best_solution, best_objective


# ============================================================================
# FPTAS FOR KNAPSACK
# ============================================================================

class FPTAS:
    """
    Fully Polynomial-Time Approximation Scheme for 0-1 Knapsack.

    The knapsack problem: maximize Σ vᵢxᵢ subject to Σ wᵢxᵢ ≤ W, xᵢ ∈ {0,1}.

    FPTAS: scale values by factor K = εV_max/n, solve scaled problem exactly
    with DP. Runs in O(n²/ε) time. Guarantees (1-ε)-approximation.
    """

    @staticmethod
    def knapsack(weights: np.ndarray, values: np.ndarray,
                 capacity: float,
                 epsilon: float = 0.1) -> tuple[list[int], float]:
        """
        FPTAS for 0-1 knapsack.

        Args:
            weights: Item weights.
            values: Item values.
            capacity: Knapsack capacity W.
            epsilon: Approximation parameter (0 < ε < 1).

        Returns:
            (items, value): Selected item indices and total value.
        """
        n = len(weights)
        if n == 0:
            return [], 0.0

        weights = np.array(weights, dtype=float)
        values = np.array(values, dtype=float)

        v_max = np.max(values)
        if v_max <= 0:
            return [], 0.0

        # Scaling factor
        K = max(epsilon * v_max / n, 1e-10)

        # Scale values and round down
        scaled_values = np.floor(values / K).astype(int)

        # DP on scaled values
        max_scaled_value = int(np.sum(scaled_values)) + 1
        # dp[v] = minimum weight to achieve exactly value v
        INF = capacity + 1
        dp = np.full(max_scaled_value, INF)
        dp[0] = 0.0

        # Track which items were used
        used = [[False] * n for _ in range(max_scaled_value)]

        for i in range(n):
            w_i = weights[i]
            v_i = scaled_values[i]

            if v_i <= 0:
                continue

            # Process in reverse to avoid using item twice
            for v in range(max_scaled_value - 1, v_i - 1, -1):
                prev_v = v - v_i
                if dp[prev_v] + w_i < dp[v]:
                    dp[v] = dp[prev_v] + w_i
                    used[v] = list(used[prev_v])
                    used[v][i] = True

        # Find best achievable value within capacity
        best_v = 0
        for v in range(max_scaled_value):
            if dp[v] <= capacity:
                best_v = v

        items = [i for i in range(n) if used[best_v][i]]
        total_value = sum(values[i] for i in items)

        logger.info(
            f"FPTAS knapsack: ε={epsilon}, "
            f"{len(items)}/{n} items, value={total_value:.2f}"
        )
        return items, total_value


# ============================================================================
# COUNT-MIN SKETCH
# ============================================================================

class Sketching:
    """
    Count-Min Sketch for frequency estimation in data streams.

    Maintains a 2D array of counters (width × depth). Each item is hashed
    by depth independent hash functions. Query returns the minimum across
    all depth rows.

    Guarantees: estimate ≤ true_count + ε·N with probability ≥ 1-δ
    where width = ⌈e/ε⌉, depth = ⌈ln(1/δ)⌉, N = total insertions.
    """

    def __init__(self, width: int, depth: int, seed: int = 42):
        """
        Args:
            width: Number of counters per row.
            depth: Number of hash functions (rows).
            seed: Random seed for hash functions.
        """
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=int)
        self.total = 0

        # Generate hash parameters (universal hashing)
        rng = np.random.default_rng(seed)
        self._a = rng.integers(1, 2 ** 31, size=depth)
        self._b = rng.integers(0, 2 ** 31, size=depth)
        self._p = 2147483647  # Large prime

    @staticmethod
    def count_sketch(data: list, width: int = 1000,
                     depth: int = 5) -> 'Sketching':
        """
        Build a Count-Min Sketch from data.

        Args:
            data: Stream of hashable items.
            width: Counter array width.
            depth: Number of hash functions.

        Returns:
            Populated Sketching object.
        """
        sketch = Sketching(width, depth)
        for item in data:
            sketch.add(item)
        return sketch

    def _hash(self, item, row: int) -> int:
        """Hash item to column index for given row."""
        h = hash(item)
        return int(((self._a[row] * h + self._b[row]) % self._p) % self.width)

    def add(self, item, count: int = 1):
        """Add item to the sketch."""
        self.total += count
        for row in range(self.depth):
            col = self._hash(item, row)
            self.table[row, col] += count

    def estimate(self, item) -> int:
        """
        Estimate frequency of item.

        Returns minimum across all hash rows (conservative estimate).
        """
        min_count = float('inf')
        for row in range(self.depth):
            col = self._hash(item, row)
            min_count = min(min_count, self.table[row, col])
        return int(min_count)

    def merge(self, other: 'Sketching') -> 'Sketching':
        """Merge two sketches (must have same dimensions and hash params)."""
        assert self.width == other.width and self.depth == other.depth
        result = Sketching(self.width, self.depth)
        result.table = self.table + other.table
        result.total = self.total + other.total
        result._a = self._a.copy()
        result._b = self._b.copy()
        return result


# ============================================================================
# MINHASH
# ============================================================================

class MinHash:
    """
    MinHash for Jaccard similarity estimation.

    The Jaccard similarity J(A, B) = |A ∩ B| / |A ∪ B| can be estimated
    by comparing MinHash signatures.

    Pr[h_min(A) = h_min(B)] = J(A, B)

    Using k independent hash functions, the estimate has standard error
    O(1/√k). With k=200, error ≈ 7%.

    Applications: near-duplicate detection, clustering, LSH.
    """

    def __init__(self, n_hashes: int = 200, seed: int = 42):
        """
        Args:
            n_hashes: Number of hash functions (signature length).
            seed: Random seed.
        """
        self.n_hashes = n_hashes
        rng = np.random.default_rng(seed)
        self._a = rng.integers(1, 2 ** 31, size=n_hashes).astype(np.int64)
        self._b = rng.integers(0, 2 ** 31, size=n_hashes).astype(np.int64)
        self._p = np.int64(2147483647)

        logger.info(f"MinHash with {n_hashes} hash functions")

    def _hash_element(self, element, i: int) -> int:
        """Hash element for the i-th hash function."""
        h = np.int64(hash(element))
        return int((self._a[i] * h + self._b[i]) % self._p)

    def signature(self, elements) -> np.ndarray:
        """
        Compute MinHash signature for a set.

        Args:
            elements: Iterable of hashable elements.

        Returns:
            Signature array of length n_hashes.
        """
        sig = np.full(self.n_hashes, np.iinfo(np.int64).max, dtype=np.int64)

        for elem in elements:
            for i in range(self.n_hashes):
                h = self._hash_element(elem, i)
                if h < sig[i]:
                    sig[i] = h

        return sig

    def jaccard_estimate(self, sig1: np.ndarray,
                         sig2: np.ndarray) -> float:
        """
        Estimate Jaccard similarity from two signatures.

        J ≈ (number of matching positions) / n_hashes
        """
        matches = np.sum(sig1 == sig2)
        return float(matches) / self.n_hashes

    def jaccard_exact(self, set1: set, set2: set) -> float:
        """Exact Jaccard similarity for comparison."""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def find_similar_pairs(self, sets: list[set],
                           threshold: float = 0.5) -> list[tuple[int, int, float]]:
        """
        Find all pairs with estimated Jaccard ≥ threshold.

        Brute force O(n² k) — for LSH-based speedup, use banding.
        """
        n = len(sets)
        sigs = [self.signature(s) for s in sets]

        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.jaccard_estimate(sigs[i], sigs[j])
                if sim >= threshold:
                    pairs.append((i, j, sim))

        logger.info(f"MinHash: {len(pairs)} similar pairs above {threshold}")
        return pairs

    def lsh_bands(self, signatures: list[np.ndarray],
                  n_bands: int = 20) -> dict:
        """
        Locality-Sensitive Hashing via banding technique.

        Split each signature into n_bands bands of r = n_hashes/n_bands rows.
        Two signatures are candidates if they agree on at least one band.

        Probability of being a candidate: 1 - (1 - J^r)^b
        where J = Jaccard, r = rows per band, b = number of bands.
        """
        r = self.n_hashes // n_bands  # Rows per band
        buckets = {}  # band -> {hash -> [signature indices]}

        for idx, sig in enumerate(signatures):
            for band in range(n_bands):
                band_key = (band, tuple(sig[band * r:(band + 1) * r]))
                if band_key not in buckets:
                    buckets[band_key] = []
                buckets[band_key].append(idx)

        # Collect candidate pairs
        candidates = set()
        for indices in buckets.values():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        candidates.add((min(indices[i], indices[j]),
                                        max(indices[i], indices[j])))

        return {'candidates': list(candidates), 'n_bands': n_bands, 'rows_per_band': r}
