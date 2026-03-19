"""
Combinatorial Numbers — Bell, Stirling, Species, Transfer Matrix
================================================================

Advanced counting structures beyond basic binomial coefficients.

Core Concepts:
- Bell numbers: Count set partitions of {1, ..., n}
- Stirling numbers: Refine Bell by number of parts (2nd kind) or cycles (1st kind)
- Combinatorial species: Joyal's theory of generating functions for structures
- Transfer matrix: Count paths in directed graphs via matrix exponentiation
- Exponential formula: Connect labeled structures to connected components

Mathematical Foundation:
B_n = Σ_{k=0}^{n} S(n,k)                     (Bell = sum of Stirling 2nd kind)
S(n,k) = k·S(n-1,k) + S(n-1,k-1)            (Stirling recurrence)
|s(n,k)| = (n-1)·|s(n-1,k)| + |s(n-1,k-1)|  (Unsigned Stirling 1st kind)
Paths of length L in digraph G: (A^L)_{ij}     (Transfer matrix method)

Applications to Warp Space:
- Partition lattice structure for memory clustering
- Cycle counting for permutation-based embeddings
- Species for structured type enumeration
- Transfer matrix for graph path analysis in knowledge graphs

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# BELL NUMBERS
# ============================================================================

class BellNumbers:
    """
    Bell numbers B_n: count the number of partitions of a set of n elements.

    B_0 = 1, B_1 = 1, B_2 = 2, B_3 = 5, B_4 = 15, B_5 = 52, ...

    Computed via the Bell triangle:
        B[0][0] = 1
        B[i][0] = B[i-1][i-1]
        B[i][j] = B[i][j-1] + B[i-1][j-1]
    """

    @staticmethod
    def compute(n: int) -> int:
        """
        Compute the n-th Bell number B_n.

        Args:
            n: Non-negative integer.

        Returns:
            B_n, the number of partitions of an n-element set.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return 1

        tri = BellNumbers.triangle(n)
        return int(tri[n - 1, -1])

    @staticmethod
    def triangle(n: int) -> np.ndarray:
        """
        Compute the Bell triangle up to row n.

        The Bell triangle has the property that the first element
        of each row gives the Bell number: B_k = tri[k][0].

        Args:
            n: Number of rows (1-indexed).

        Returns:
            Array of shape (n, n) with the Bell triangle.
        """
        if n <= 0:
            return np.array([[1]])

        tri = np.zeros((n, n), dtype=np.int64)
        tri[0, 0] = 1

        for i in range(1, n):
            tri[i, 0] = tri[i - 1, i - 1]
            for j in range(1, i + 1):
                tri[i, j] = tri[i, j - 1] + tri[i - 1, j - 1]

        return tri

    @staticmethod
    def sequence(n: int) -> np.ndarray:
        """Return B_0, B_1, ..., B_n as an array."""
        if n < 0:
            return np.array([], dtype=np.int64)

        result = np.zeros(n + 1, dtype=np.int64)
        result[0] = 1
        if n == 0:
            return result

        tri = BellNumbers.triangle(n)
        for k in range(1, n + 1):
            result[k] = tri[k - 1, 0]
        # Fix: B_k is the last element of row k-1
        # Actually B_k = tri[k-1][-1] for the standard construction
        # Let's recompute properly
        result[0] = 1
        for k in range(1, n + 1):
            result[k] = tri[k - 1, k - 1]

        return result


# ============================================================================
# STIRLING NUMBERS
# ============================================================================

class StirlingNumbers:
    """
    Stirling numbers of the first and second kind.

    First kind |s(n,k)|: number of permutations of n elements with exactly k cycles.
    Second kind S(n,k): number of ways to partition n elements into exactly k
    non-empty subsets.

    Both satisfy B_n = Σ_k S(n,k) (Bell number decomposition).
    """

    @staticmethod
    @lru_cache(maxsize=4096)
    def first_kind(n: int, k: int) -> int:
        """
        Unsigned Stirling number of the first kind |s(n,k)|.

        Counts permutations of n elements with exactly k disjoint cycles.

        Recurrence: |s(n,k)| = (n-1)·|s(n-1,k)| + |s(n-1,k-1)|
        Base cases: |s(0,0)| = 1, |s(n,0)| = 0 for n>0, |s(0,k)| = 0 for k>0.
        """
        if n < 0 or k < 0:
            return 0
        if n == 0 and k == 0:
            return 1
        if n == 0 or k == 0:
            return 0
        if k > n:
            return 0
        return (n - 1) * StirlingNumbers.first_kind(n - 1, k) + \
            StirlingNumbers.first_kind(n - 1, k - 1)

    @staticmethod
    @lru_cache(maxsize=4096)
    def second_kind(n: int, k: int) -> int:
        """
        Stirling number of the second kind S(n,k).

        Counts partitions of {1,...,n} into exactly k non-empty subsets.

        Recurrence: S(n,k) = k·S(n-1,k) + S(n-1,k-1)
        Base cases: S(0,0) = 1, S(n,0) = 0 for n>0, S(0,k) = 0 for k>0.
        """
        if n < 0 or k < 0:
            return 0
        if n == 0 and k == 0:
            return 1
        if n == 0 or k == 0:
            return 0
        if k > n:
            return 0
        return k * StirlingNumbers.second_kind(n - 1, k) + \
            StirlingNumbers.second_kind(n - 1, k - 1)

    @staticmethod
    def table_first(n: int) -> np.ndarray:
        """Build table of |s(i,j)| for 0 <= i,j <= n."""
        T = np.zeros((n + 1, n + 1), dtype=np.int64)
        for i in range(n + 1):
            for j in range(i + 1):
                T[i, j] = StirlingNumbers.first_kind(i, j)
        return T

    @staticmethod
    def table_second(n: int) -> np.ndarray:
        """Build table of S(i,j) for 0 <= i,j <= n."""
        T = np.zeros((n + 1, n + 1), dtype=np.int64)
        for i in range(n + 1):
            for j in range(i + 1):
                T[i, j] = StirlingNumbers.second_kind(i, j)
        return T


# ============================================================================
# COMBINATORIAL SPECIES
# ============================================================================

class Species:
    """
    Joyal's theory of combinatorial species.

    A species F is a functor from finite sets to finite sets. Its generating
    function encodes the number of F-structures on sets of each size.

    Type generating function: F(x) = Σ_n f_n x^n / n!
    where f_n = |F[n]| is the number of F-structures on an n-element set.
    """

    # Known species and their counts f_n
    SPECIES_TABLE = {
        'set': lambda n: 1,                          # One set structure per n
        'permutation': lambda n: int(np.math.factorial(n)),  # n! permutations
        'linear_order': lambda n: int(np.math.factorial(n)),
        'cycle': lambda n: int(np.math.factorial(n - 1)) if n >= 1 else 0,
        'subset': lambda n: 2 ** n,                  # Power set
        'partition': lambda n: BellNumbers.compute(n),
        'tree': lambda n: int(n ** (n - 1)) if n >= 1 else (1 if n == 0 else 0),  # Cayley
        'derangement': lambda n: Species._derangement_count(n),
        'involution': lambda n: Species._involution_count(n),
    }

    @staticmethod
    def _derangement_count(n: int) -> int:
        """D_n = n! Σ_{k=0}^{n} (-1)^k / k!"""
        if n == 0:
            return 1
        total = 0.0
        for k in range(n + 1):
            total += ((-1) ** k) / np.math.factorial(k)
        return int(round(np.math.factorial(n) * total))

    @staticmethod
    def _involution_count(n: int) -> int:
        """a(n) = a(n-1) + (n-1)*a(n-2), a(0)=a(1)=1."""
        if n <= 1:
            return 1
        a, b = 1, 1  # a(0), a(1)
        for k in range(2, n + 1):
            a, b = b, b + (k - 1) * a
        return b

    @staticmethod
    def generating_function(species_type: str, n: int) -> float:
        """
        Evaluate the n-th coefficient of the species' EGF.

        Returns f_n / n! (the EGF coefficient), or f_n (the count) depending
        on context. Returns the raw count f_n.

        Args:
            species_type: Key from SPECIES_TABLE.
            n: Size of the underlying set.

        Returns:
            f_n, the number of species_type-structures on an n-element set.
        """
        if species_type not in Species.SPECIES_TABLE:
            raise ValueError(
                f"Unknown species '{species_type}'. "
                f"Known: {list(Species.SPECIES_TABLE.keys())}"
            )
        return float(Species.SPECIES_TABLE[species_type](n))


# ============================================================================
# TRANSFER MATRIX
# ============================================================================

class TransferMatrix:
    """
    Path counting in directed graphs via matrix exponentiation.

    The number of walks of length L from vertex i to vertex j in a
    directed graph with adjacency matrix A is (A^L)_{ij}.

    This is the transfer matrix method, fundamental in combinatorics
    and statistical mechanics.
    """

    @staticmethod
    def count_paths(adjacency: np.ndarray, length: int) -> int:
        """
        Count total number of walks of given length in the graph.

        Args:
            adjacency: Square adjacency matrix (n x n).
            length: Walk length L >= 0.

        Returns:
            Total number of walks of length L (sum of all entries in A^L).
        """
        if length < 0:
            raise ValueError("Walk length must be non-negative")
        n = adjacency.shape[0]
        if adjacency.shape != (n, n):
            raise ValueError("Adjacency matrix must be square")

        power = np.linalg.matrix_power(adjacency.astype(float), length)
        return int(round(np.sum(power)))

    @staticmethod
    def count_paths_ij(adjacency: np.ndarray, length: int,
                       i: int, j: int) -> int:
        """Count walks of given length from vertex i to vertex j."""
        n = adjacency.shape[0]
        power = np.linalg.matrix_power(adjacency.astype(float), length)
        return int(round(power[i, j]))

    @staticmethod
    def path_matrix(adjacency: np.ndarray, length: int) -> np.ndarray:
        """
        Full path count matrix: (A^L)_{ij} = walks of length L from i to j.
        """
        return np.linalg.matrix_power(adjacency.astype(float), length)

    @staticmethod
    def total_paths_up_to(adjacency: np.ndarray, max_length: int) -> int:
        """
        Count all walks of length 0, 1, ..., max_length.

        Uses the identity: Σ_{k=0}^{L} A^k = (I - A^{L+1})(I - A)^{-1}
        when (I - A) is invertible, else iterative summation.
        """
        n = adjacency.shape[0]
        A = adjacency.astype(float)
        total = np.zeros((n, n))
        current = np.eye(n)

        for _ in range(max_length + 1):
            total += current
            current = current @ A

        return int(round(np.sum(total)))


# ============================================================================
# EXPONENTIAL FORMULA
# ============================================================================

class ExponentialFormula:
    """
    The exponential formula in combinatorics.

    If C(x) = Σ c_n x^n/n! is the EGF for connected structures, then
    A(x) = exp(C(x)) is the EGF for all structures (possibly disconnected).

    Equivalently: a_n = Σ over partitions of n into parts, product of
    c_{part_size} terms.

    This relates connected and labeled structures via:
        A(x) = exp(C(x))  ⟺  C(x) = log(A(x))
    """

    @staticmethod
    def connected_to_labeled(connected_gf: np.ndarray, n: int) -> float:
        """
        Given EGF coefficients of connected structures, compute the
        n-th coefficient of the EGF for all (possibly disconnected) structures.

        Uses: A(x) = exp(C(x)), computed via the recurrence:
            a_0 = exp(c_0)
            n * a_n = Σ_{k=1}^{n} k * c_k * a_{n-k}

        Args:
            connected_gf: Array of c_0, c_1, ..., c_m (EGF coefficients
                          of connected structures, i.e., c_n / n!).
            n: Index to compute.

        Returns:
            a_n, the n-th EGF coefficient of all structures.
        """
        c = np.zeros(n + 1)
        c[:min(len(connected_gf), n + 1)] = connected_gf[:min(len(connected_gf), n + 1)]

        a = np.zeros(n + 1)
        a[0] = np.exp(c[0])

        for i in range(1, n + 1):
            s = 0.0
            for k in range(1, i + 1):
                if k < len(c):
                    s += k * c[k] * a[i - k]
            a[i] = s / i

        return float(a[n])

    @staticmethod
    def labeled_to_connected(labeled_gf: np.ndarray, n: int) -> float:
        """
        Inverse: given EGF of all structures, recover connected component EGF.

        Uses: C(x) = log(A(x)), computed via:
            c_0 = log(a_0)
            n * c_n = n * a_n - Σ_{k=1}^{n-1} k * c_k * a_{n-k}

        Args:
            labeled_gf: Array of a_0, a_1, ..., a_m (EGF coefficients).
            n: Index to compute.

        Returns:
            c_n, the n-th EGF coefficient of connected structures.
        """
        a = np.zeros(n + 1)
        a[:min(len(labeled_gf), n + 1)] = labeled_gf[:min(len(labeled_gf), n + 1)]

        if a[0] <= 0:
            raise ValueError("a_0 must be positive for log(A(x))")

        c = np.zeros(n + 1)
        c[0] = np.log(a[0])

        for i in range(1, n + 1):
            s = 0.0
            for k in range(1, i):
                s += k * c[k] * a[i - k]
            c[i] = (i * a[i] - s) / i

        return float(c[n])
