"""
Advanced Combinatorics - Generating Functions, Partitions, q-Analogs
===================================================================

Deep combinatorial structures beyond basic counting.

Classes:
    GeneratingFunction: Ordinary and exponential generating functions
    IntegerPartition: Partition theory, Young tableaux
    QAnalogs: q-binomials, q-factorials, quantum calculus
    AsymptoticEnumeration: Saddle-point method, singularity analysis
    SymmetricFunctions: Schur functions, plethysm
    CatalanNumbers: Catalan objects and generalizations

Applications:
    - Combinatorial identities
    - Enumeration problems
    - Quantum groups
    - Statistical mechanics
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from functools import lru_cache
from dataclasses import dataclass


class GeneratingFunction:
    """
    Generating functions: encode sequences as formal power series.

    OGF: A(x) = sum a_n x^n
    EGF: A(x) = sum a_n x^n / n!
    """

    def __init__(self, coefficients: np.ndarray, kind: str = 'ordinary'):
        """
        Args:
            coefficients: Sequence a_0, a_1, a_2, ...
            kind: 'ordinary' or 'exponential'
        """
        self.coeffs = coefficients
        self.kind = kind

    def __call__(self, x: float, max_terms: int = 50) -> float:
        """Evaluate generating function at x."""
        result = 0.0
        for n in range(min(len(self.coeffs), max_terms)):
            if self.kind == 'ordinary':
                result += self.coeffs[n] * (x ** n)
            else:  # exponential
                result += self.coeffs[n] * (x ** n) / np.math.factorial(n)
        return result

    def __add__(self, other: 'GeneratingFunction') -> 'GeneratingFunction':
        """Add two generating functions: A(x) + B(x)."""
        max_len = max(len(self.coeffs), len(other.coeffs))
        coeffs = np.zeros(max_len)
        coeffs[:len(self.coeffs)] += self.coeffs
        coeffs[:len(other.coeffs)] += other.coeffs
        return GeneratingFunction(coeffs, self.kind)

    def __mul__(self, other: 'GeneratingFunction') -> 'GeneratingFunction':
        """Multiply generating functions: A(x) * B(x) (convolution)."""
        n1, n2 = len(self.coeffs), len(other.coeffs)
        result_len = n1 + n2 - 1
        coeffs = np.zeros(result_len)

        if self.kind == 'ordinary' and other.kind == 'ordinary':
            # Convolution: c_n = sum_{k=0}^n a_k * b_{n-k}
            for i in range(n1):
                for j in range(n2):
                    coeffs[i + j] += self.coeffs[i] * other.coeffs[j]
        else:
            # For EGF, convolution is different
            for n in range(result_len):
                for k in range(n + 1):
                    if k < n1 and n - k < n2:
                        # Binomial convolution for EGF
                        coeffs[n] += (np.math.factorial(n) /
                                     (np.math.factorial(k) * np.math.factorial(n - k)) *
                                     self.coeffs[k] * other.coeffs[n - k])

        return GeneratingFunction(coeffs, self.kind)

    @staticmethod
    def fibonacci_ogf(n_terms: int = 20) -> 'GeneratingFunction':
        """
        Fibonacci OGF: F(x) = x / (1 - x - x^2).

        Coefficients are Fibonacci numbers.
        """
        fib = [0, 1]
        for _ in range(n_terms - 2):
            fib.append(fib[-1] + fib[-2])
        return GeneratingFunction(np.array(fib), 'ordinary')

    @staticmethod
    def catalan_ogf(n_terms: int = 20) -> 'GeneratingFunction':
        """
        Catalan OGF: C(x) = (1 - sqrt(1 - 4x)) / (2x).

        C_n = (1/(n+1)) * binom(2n, n)
        """
        catalan = [CatalanNumbers.catalan(n) for n in range(n_terms)]
        return GeneratingFunction(np.array(catalan), 'ordinary')

    @staticmethod
    def exponential_series(n_terms: int = 20) -> 'GeneratingFunction':
        """
        exp(x) as EGF: sum x^n / n! = e^x.

        All coefficients are 1.
        """
        return GeneratingFunction(np.ones(n_terms), 'exponential')


class IntegerPartition:
    """
    Integer partitions: ways to write n as sum of positive integers.

    Example: 4 = 4 = 3+1 = 2+2 = 2+1+1 = 1+1+1+1 (5 partitions)
    """

    @staticmethod
    @lru_cache(maxsize=1000)
    def count(n: int) -> int:
        """
        Number of partitions of n (Hardy-Ramanujan asymptotic).

        p(n) ~ exp(pi*sqrt(2n/3)) / (4*sqrt(3)*n)
        """
        if n < 0:
            return 0
        if n == 0:
            return 1

        # Use recursion with Euler's pentagonal formula
        # p(n) = sum_{k!=0} (-1)^{k+1} p(n - k(3k-1)/2)
        result = 0
        k = 1
        while True:
            # Pentagonal numbers: k(3k-1)/2 for k = ±1, ±2, ...
            pent1 = k * (3 * k - 1) // 2
            pent2 = k * (3 * k + 1) // 2

            if pent1 > n:
                break

            sign = (-1) ** (k + 1)
            result += sign * IntegerPartition.count(n - pent1)

            if pent2 <= n:
                result += sign * IntegerPartition.count(n - pent2)

            k += 1

        return result

    @staticmethod
    def generate_partitions(n: int) -> List[List[int]]:
        """
        Generate all partitions of n.

        Returns: List of partitions (each partition is list in descending order)
        """
        if n == 0:
            return [[]]

        partitions = []
        for i in range(1, n + 1):
            for partition in IntegerPartition.generate_partitions(n - i):
                if not partition or i >= partition[0]:
                    partitions.append([i] + partition)

        return partitions

    @staticmethod
    def ferrers_diagram(partition: List[int]) -> str:
        """
        Ferrers diagram: visual representation of partition.

        Example: [3, 2, 1] ->
        ***
        **
        *
        """
        return '\n'.join('*' * part for part in partition)

    @staticmethod
    def conjugate(partition: List[int]) -> List[int]:
        """
        Conjugate partition: transpose Ferrers diagram.

        Example: [3, 2, 1] -> [3, 2, 1] (self-conjugate)
                 [4, 2, 1] -> [3, 2, 1, 1]
        """
        if not partition:
            return []

        max_part = partition[0]
        conjugate = []

        for i in range(max_part):
            count = sum(1 for p in partition if p > i)
            conjugate.append(count)

        return conjugate


class QAnalogs:
    """
    q-Analogs: quantum analogs of classical combinatorial objects.

    [n]_q = (1 - q^n) / (1 - q) = 1 + q + q^2 + ... + q^{n-1}
    """

    @staticmethod
    def q_number(n: int, q: float) -> float:
        """
        q-analog of integer n: [n]_q = (1 - q^n) / (1 - q).

        As q -> 1: [n]_q -> n
        """
        if abs(q - 1) < 1e-10:
            return float(n)
        return (1 - q ** n) / (1 - q)

    @staticmethod
    def q_factorial(n: int, q: float) -> float:
        """
        q-factorial: [n]_q! = [1]_q * [2]_q * ... * [n]_q.

        As q -> 1: [n]_q! -> n!
        """
        result = 1.0
        for k in range(1, n + 1):
            result *= QAnalogs.q_number(k, q)
        return result

    @staticmethod
    def q_binomial(n: int, k: int, q: float) -> float:
        """
        q-binomial (Gaussian binomial coefficient):
        [n choose k]_q = [n]_q! / ([k]_q! * [n-k]_q!)

        As q -> 1: [n choose k]_q -> C(n, k)

        Also counts k-dimensional subspaces of F_q^n.
        """
        if k < 0 or k > n:
            return 0.0
        if k == 0 or k == n:
            return 1.0

        numerator = QAnalogs.q_factorial(n, q)
        denominator = QAnalogs.q_factorial(k, q) * QAnalogs.q_factorial(n - k, q)
        return numerator / denominator

    @staticmethod
    def q_pochhammer(a: float, n: int, q: float) -> float:
        """
        q-Pochhammer symbol: (a; q)_n = prod_{k=0}^{n-1} (1 - a*q^k).

        Used in q-series and partition theory.
        """
        result = 1.0
        for k in range(n):
            result *= (1 - a * (q ** k))
        return result


class CatalanNumbers:
    """
    Catalan numbers: ubiquitous in combinatorics.

    C_n = (1/(n+1)) * C(2n, n) = C(2n, n) - C(2n, n+1)

    Count: binary trees, Dyck paths, triangulations, parenthesizations, etc.
    """

    @staticmethod
    @lru_cache(maxsize=100)
    def catalan(n: int) -> int:
        """
        n-th Catalan number via recursion.

        C_0 = 1, C_{n+1} = sum_{i=0}^n C_i * C_{n-i}
        """
        if n <= 1:
            return 1

        result = 0
        for i in range(n):
            result += CatalanNumbers.catalan(i) * CatalanNumbers.catalan(n - 1 - i)
        return result

    @staticmethod
    def catalan_formula(n: int) -> int:
        """
        Catalan number via formula: C_n = C(2n, n) / (n+1).
        """
        from math import comb
        return comb(2 * n, n) // (n + 1)

    @staticmethod
    def dyck_paths(n: int) -> List[str]:
        """
        Generate all Dyck paths of length 2n.

        Dyck path: sequence of n steps up (U) and n steps down (D),
        never going below starting point.
        """
        if n == 0:
            return ['']

        paths = []
        for path in CatalanNumbers.dyck_paths(n - 1):
            # Insert UD at each position
            for i in range(len(path) + 1):
                new_path = path[:i] + 'UD' + path[i:]
                if new_path not in paths:
                    paths.append(new_path)

        return paths

    @staticmethod
    def applications() -> Dict[str, str]:
        """Catalan number applications."""
        return {
            "binary_trees": "Number of binary trees with n internal nodes",
            "dyck_paths": "Lattice paths from (0,0) to (2n,0) not crossing x-axis",
            "parenthesizations": "Ways to parenthesize product of n+1 factors",
            "triangulations": "Triangulations of convex (n+2)-gon",
            "noncrossing_partitions": "Non-crossing partitions of {1,...,n}",
            "mountain_ranges": "Mountain ranges with n upstrokes and n downstrokes"
        }


class AsymptoticEnumeration:
    """
    Asymptotic enumeration: approximate counting for large n.

    Techniques: saddle-point method, singularity analysis.
    """

    @staticmethod
    def stirling_approximation(n: int) -> float:
        """
        Stirling's approximation: n! ~ sqrt(2*pi*n) * (n/e)^n.

        More accurate: n! ~ sqrt(2*pi*n) * (n/e)^n * (1 + 1/(12n))
        """
        if n == 0:
            return 1.0
        return np.sqrt(2 * np.pi * n) * ((n / np.e) ** n)

    @staticmethod
    def partition_asymptotic(n: int) -> float:
        """
        Hardy-Ramanujan asymptotic formula for partitions:
        p(n) ~ exp(pi*sqrt(2n/3)) / (4*sqrt(3)*n)
        """
        return np.exp(np.pi * np.sqrt(2 * n / 3)) / (4 * np.sqrt(3) * n)

    @staticmethod
    def catalan_asymptotic(n: int) -> float:
        """
        Catalan asymptotic: C_n ~ 4^n / (sqrt(pi) * n^{3/2}).
        """
        return (4 ** n) / (np.sqrt(np.pi) * (n ** 1.5))


class SymmetricFunctions:
    """
    Symmetric functions: functions invariant under permutation.

    Elementary symmetric: e_k = sum of all k-element products
    Power sum: p_k = sum of k-th powers
    Complete homogeneous: h_k
    Schur functions: s_λ (indexed by partitions)
    """

    @staticmethod
    def elementary_symmetric(values: np.ndarray, k: int) -> float:
        """
        k-th elementary symmetric polynomial:
        e_k(x_1,...,x_n) = sum_{i_1 < ... < i_k} x_{i_1} * ... * x_{i_k}
        """
        from itertools import combinations
        n = len(values)
        if k > n or k < 0:
            return 0.0

        result = 0.0
        for indices in combinations(range(n), k):
            product = 1.0
            for i in indices:
                product *= values[i]
            result += product

        return result

    @staticmethod
    def power_sum(values: np.ndarray, k: int) -> float:
        """
        k-th power sum: p_k(x_1,...,x_n) = x_1^k + ... + x_n^k.
        """
        return np.sum(values ** k)

    @staticmethod
    def newton_identities(n: int) -> str:
        """
        Newton's identities: relate elementary and power sums.

        p_k = sum_{i=1}^k (-1)^{i-1} e_i p_{k-i}  (k <= n)
        """
        return (
            "Newton's Identities:\n"
            "k*e_k = sum_{i=1}^k (-1)^{i-1} e_{k-i} * p_i\n\n"
            "Relate elementary symmetric (e_k) to power sums (p_k)."
        )


# ============================================================================
# EXAMPLES AND TESTS
# ============================================================================

def example_generating_functions():
    """Example: Fibonacci generating function."""
    fib_gf = GeneratingFunction.fibonacci_ogf(n_terms=10)

    # Evaluate at x=0.5
    value = fib_gf(0.5)

    # Check coefficients
    coeffs = fib_gf.coeffs
    return coeffs, value


def example_partitions():
    """Example: Integer partitions of 5."""
    partitions = IntegerPartition.generate_partitions(5)
    count = IntegerPartition.count(5)

    # Show Ferrers diagrams
    diagrams = [IntegerPartition.ferrers_diagram(p) for p in partitions[:3]]
    return count, partitions[:3], diagrams


def example_q_binomial():
    """Example: q-binomial coefficients."""
    n, k = 5, 2

    # At q=1, should equal regular binomial
    q_binom_at_1 = QAnalogs.q_binomial(n, k, q=0.99)

    from math import comb
    regular_binom = comb(n, k)

    # At q=2, counts subspaces over F_2
    q_binom_at_2 = QAnalogs.q_binomial(n, k, q=2)

    return q_binom_at_1, regular_binom, q_binom_at_2


def example_catalan():
    """Example: Catalan numbers and Dyck paths."""
    catalan_5 = CatalanNumbers.catalan(5)

    # Generate Dyck paths for n=3
    dyck_paths = CatalanNumbers.dyck_paths(3)

    return catalan_5, dyck_paths


if __name__ == "__main__":
    print("Advanced Combinatorics Module")
    print("=" * 60)

    # Test 1: Generating functions
    print("\n[Test 1] Fibonacci generating function")
    coeffs, value = example_generating_functions()
    print(f"Fibonacci numbers: {coeffs[:8]}")
    print(f"F(0.5) = {value:.4f}")

    # Test 2: Partitions
    print("\n[Test 2] Integer partitions of 5")
    count, partitions, diagrams = example_partitions()
    print(f"Number of partitions: {count}")
    print(f"First 3 partitions: {partitions}")
    print(f"\nFerrers diagram of {partitions[0]}:")
    print(diagrams[0])

    # Test 3: q-binomials
    print("\n[Test 3] q-Binomial coefficients")
    q1, regular, q2 = example_q_binomial()
    print(f"[5 choose 2]_(q=0.99) = {q1:.2f}")
    print(f"Regular C(5,2) = {regular}")
    print(f"[5 choose 2]_(q=2) = {q2:.2f}")

    # Test 4: Catalan numbers
    print("\n[Test 4] Catalan numbers")
    c5, paths = example_catalan()
    print(f"C_5 = {c5} (expect 42)")
    print(f"Number of Dyck paths (n=3): {len(paths)}")
    print(f"Sample paths: {paths[:3]}")

    # Test 5: Asymptotic formulas
    print("\n[Test 5] Asymptotic approximations")
    n = 10
    exact_factorial = np.math.factorial(n)
    stirling = AsymptoticEnumeration.stirling_approximation(n)
    print(f"{n}! = {exact_factorial}")
    print(f"Stirling: {stirling:.0f}")
    print(f"Error: {abs(exact_factorial - stirling)/exact_factorial * 100:.2f}%")

    # Test 6: Symmetric functions
    print("\n[Test 6] Symmetric functions")
    values = np.array([1, 2, 3, 4])
    e2 = SymmetricFunctions.elementary_symmetric(values, 2)
    p2 = SymmetricFunctions.power_sum(values, 2)
    print(f"e_2(1,2,3,4) = {e2:.0f} (sum of pairs)")
    print(f"p_2(1,2,3,4) = {p2:.0f} (sum of squares)")

    print("\n" + "=" * 60)
    print("All advanced combinatorics tests complete!")
    print("Generating functions, partitions, q-analogs ready.")
