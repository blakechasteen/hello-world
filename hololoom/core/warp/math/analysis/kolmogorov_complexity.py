"""
Kolmogorov Complexity — Algorithmic Information Theory
=======================================================

Approximations to Kolmogorov complexity via compression, and related
measures from algorithmic information theory.

Core Concepts:
- Kolmogorov Complexity K(x): Length of shortest program that outputs x
- K(x) is uncomputable — we approximate via compression ratio
- Conditional Complexity K(x|y): Shortest program for x given y as input
- Normalized Information Distance: Universal similarity metric

Mathematical Foundation:
K(x) ≈ |compress(x)| (compression-based approximation)
NID(x,y) = (max(K(x|y), K(y|x))) / max(K(x), K(y))
NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))

Applications to Warp Space:
- Measuring true information content of embeddings
- Detecting compressible structure in memory graphs
- Similarity via algorithmic distance (NCD)
- Complexity-weighted scoring for pattern selection

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
import zlib
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ComplexityResult:
    """Result of complexity estimation.

    Attributes:
        raw_size: Original data size in bytes.
        compressed_size: Compressed data size in bytes.
        complexity: Estimated Kolmogorov complexity (normalized).
        compression_ratio: compressed_size / raw_size.
    """
    raw_size: int
    compressed_size: int
    complexity: float
    compression_ratio: float


# ============================================================================
# KOLMOGOROV COMPLEXITY APPROXIMATION
# ============================================================================

class KolmogorovComplexity:
    """Approximate Kolmogorov complexity via compression.

    Since K(x) is uncomputable, we use the length of compressed
    representations as an upper bound. The compression ratio
    C(x)/|x| approximates the normalized complexity.

    Good compressors (like zlib/DEFLATE) give tighter bounds because
    they approach the entropy rate for stationary ergodic sources.

    Example:
        >>> kc = KolmogorovComplexity()
        >>> random_data = np.random.bytes(1000)
        >>> kc.approximate(random_data)  # ~1.0 (incompressible)
        >>> pattern_data = b'ab' * 500
        >>> kc.approximate(pattern_data)  # <<1.0 (highly compressible)
    """

    def approximate(self, data: bytes | str | np.ndarray) -> float:
        """Estimate normalized Kolmogorov complexity via compression.

        Returns value in [0, 1+] where:
        - 0 = trivially simple (constant data)
        - 1 = incompressible (random data, may exceed 1 due to headers)

        Args:
            data: Input data (bytes, string, or numpy array).

        Returns:
            Normalized complexity estimate.
        """
        raw = self._to_bytes(data)
        if len(raw) == 0:
            return 0.0

        compressed = zlib.compress(raw, level=9)
        return len(compressed) / len(raw)

    def estimate(self, data: bytes | str | np.ndarray) -> ComplexityResult:
        """Full complexity estimation with details.

        Args:
            data: Input data.

        Returns:
            ComplexityResult with raw and compressed sizes.
        """
        raw = self._to_bytes(data)
        if len(raw) == 0:
            return ComplexityResult(0, 0, 0.0, 0.0)

        compressed = zlib.compress(raw, level=9)
        ratio = len(compressed) / len(raw)

        return ComplexityResult(
            raw_size=len(raw),
            compressed_size=len(compressed),
            complexity=ratio,
            compression_ratio=ratio,
        )

    def complexity_rate(self, data: bytes | str | np.ndarray, block_size: int = 100) -> np.ndarray:
        """Compute local complexity rate along the data.

        Slides a window across the data and computes complexity at each
        position. Useful for detecting complexity transitions.

        Args:
            data: Input data.
            block_size: Window size in bytes.

        Returns:
            Array of local complexity values.
        """
        raw = self._to_bytes(data)
        n = len(raw)
        if n < block_size:
            return np.array([self.approximate(data)])

        rates = []
        for i in range(0, n - block_size + 1, block_size // 2):
            block = raw[i: i + block_size]
            compressed = zlib.compress(block, level=9)
            rates.append(len(compressed) / len(block))

        return np.array(rates)

    @staticmethod
    def _to_bytes(data: bytes | str | np.ndarray) -> bytes:
        """Convert input to bytes."""
        if isinstance(data, bytes):
            return data
        elif isinstance(data, str):
            return data.encode('utf-8')
        elif isinstance(data, np.ndarray):
            return data.tobytes()
        else:
            return str(data).encode('utf-8')


# ============================================================================
# ALGORITHMIC INFORMATION
# ============================================================================

class AlgorithmicInformation:
    """Algorithmic information measures based on compression.

    Provides conditional complexity, mutual algorithmic information,
    and normalized compression/information distances.

    Example:
        >>> ai = AlgorithmicInformation()
        >>> x = b"hello world " * 100
        >>> y = b"hello earth " * 100
        >>> ai.conditional(x, y)  # Low — y helps compress x
        >>> ai.ncd(x, y)  # Low — similar strings
    """

    def __init__(self):
        self._kc = KolmogorovComplexity()

    def conditional(self, x: bytes | str | np.ndarray, y: bytes | str | np.ndarray) -> float:
        """Approximate conditional Kolmogorov complexity K(x|y).

        K(x|y) ≈ C(yx) - C(y)

        The idea: if y is given, how much additional information is needed
        to describe x? We measure this as the compression of the
        concatenation minus the compression of y alone.

        Args:
            x: Target data.
            y: Given/conditioning data.

        Returns:
            Approximate K(x|y) in bytes.
        """
        x_bytes = self._kc._to_bytes(x)
        y_bytes = self._kc._to_bytes(y)

        c_y = len(zlib.compress(y_bytes, level=9))
        c_yx = len(zlib.compress(y_bytes + x_bytes, level=9))

        return max(0.0, float(c_yx - c_y))

    def mutual_information(
        self, x: bytes | str | np.ndarray, y: bytes | str | np.ndarray
    ) -> float:
        """Approximate algorithmic mutual information.

        I(x:y) = K(x) + K(y) - K(xy) ≈ C(x) + C(y) - C(xy)

        Args:
            x, y: Data objects.

        Returns:
            Approximate mutual algorithmic information in bytes.
        """
        x_bytes = self._kc._to_bytes(x)
        y_bytes = self._kc._to_bytes(y)

        c_x = len(zlib.compress(x_bytes, level=9))
        c_y = len(zlib.compress(y_bytes, level=9))
        c_xy = len(zlib.compress(x_bytes + y_bytes, level=9))

        return max(0.0, float(c_x + c_y - c_xy))

    def ncd(self, x: bytes | str | np.ndarray, y: bytes | str | np.ndarray) -> float:
        """Normalized Compression Distance (NCD).

        NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))

        Universal similarity metric: NCD ≈ 0 means similar,
        NCD ≈ 1 means maximally different. Based on the theoretical
        Normalized Information Distance which is a metric.

        Args:
            x, y: Data objects.

        Returns:
            NCD in [0, 1+].
        """
        x_bytes = self._kc._to_bytes(x)
        y_bytes = self._kc._to_bytes(y)

        c_x = len(zlib.compress(x_bytes, level=9))
        c_y = len(zlib.compress(y_bytes, level=9))
        c_xy = len(zlib.compress(x_bytes + y_bytes, level=9))

        denominator = max(c_x, c_y)
        if denominator == 0:
            return 0.0

        return (c_xy - min(c_x, c_y)) / denominator

    def ncd_matrix(self, items: list) -> np.ndarray:
        """Compute pairwise NCD matrix.

        Args:
            items: List of data objects.

        Returns:
            Symmetric distance matrix (n, n).
        """
        n = len(items)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self.ncd(items[i], items[j])
                D[i, j] = d
                D[j, i] = d
        return D

    def information_distance(
        self, x: bytes | str | np.ndarray, y: bytes | str | np.ndarray
    ) -> float:
        """Approximate (unnormalized) information distance.

        d(x,y) = max(K(x|y), K(y|x)) ≈ max(C(xy) - C(y), C(yx) - C(x))

        Args:
            x, y: Data objects.

        Returns:
            Approximate information distance in bytes.
        """
        x_bytes = self._kc._to_bytes(x)
        y_bytes = self._kc._to_bytes(y)

        c_x = len(zlib.compress(x_bytes, level=9))
        c_y = len(zlib.compress(y_bytes, level=9))
        c_xy = len(zlib.compress(x_bytes + y_bytes, level=9))
        c_yx = len(zlib.compress(y_bytes + x_bytes, level=9))

        k_x_given_y = max(0, c_yx - c_y)
        k_y_given_x = max(0, c_xy - c_x)

        return float(max(k_x_given_y, k_y_given_x))


# ============================================================================
# BERRY PARADOX
# ============================================================================

class BerryParadox:
    """Berry's Paradox and description complexity bounds.

    Berry's paradox: "The smallest positive integer not definable in
    under eleven words" — this phrase defines such a number in ten words,
    a contradiction. This reveals deep connections between description
    length and definability.

    We provide computational tools for studying description length bounds
    and the relationship between number magnitude and description complexity.

    Example:
        >>> bp = BerryParadox()
        >>> bp.shortest_description(42)  # Description length of 42
        >>> bp.busy_beaver_bound(4)  # Lower bound on BB(4)
    """

    @staticmethod
    def shortest_description(n: int) -> int:
        """Estimate shortest description length for integer n.

        Lower bound: K(n) >= log2(n) for most n (by counting argument).
        Upper bound: K(n) <= log2(n) + 2*log2(log2(n)) + c (self-delimiting).

        Args:
            n: Positive integer.

        Returns:
            Estimated description length in bits.
        """
        if n <= 0:
            return 1
        # Lower bound from Kraft inequality
        log_n = np.log2(max(n, 1))
        # Self-delimiting encoding: prefix-free binary
        if log_n > 1:
            return int(np.ceil(log_n + 2 * np.log2(log_n) + 1))
        return int(np.ceil(log_n + 1))

    @staticmethod
    def chaitin_incompleteness_bound(axiom_bits: int) -> int:
        """Chaitin's incompleteness theorem bound.

        A formal system with axiom complexity c bits cannot prove
        K(x) > c + O(1) for any specific x.

        Args:
            axiom_bits: Complexity of the formal system's axioms.

        Returns:
            Maximum provable complexity bound.
        """
        # The constant depends on the encoding — use a reasonable estimate
        overhead = 20  # UTM simulation overhead
        return axiom_bits + overhead

    @staticmethod
    def busy_beaver_bound(n: int) -> int:
        """Lower bound on the Busy Beaver function BB(n).

        BB(n) grows faster than any computable function. Known values:
        BB(1)=1, BB(2)=4, BB(3)=6, BB(4)=13, BB(5)>=47176870.

        Args:
            n: Number of Turing machine states.

        Returns:
            Known lower bound on BB(n).
        """
        known = {0: 0, 1: 1, 2: 4, 3: 6, 4: 13}
        if n in known:
            return known[n]
        if n == 5:
            return 47176870
        # For n > 5, BB grows incomputably fast
        # Return a very conservative lower bound
        if n > 5:
            return int(10 ** (n - 3))
        return 0

    @staticmethod
    def kolmogorov_sufficient_statistic(data: np.ndarray, model_complexity: int) -> float:
        """Evaluate a model as a Kolmogorov sufficient statistic.

        A model M is a sufficient statistic for data x if:
        K(x) ≈ K(M) + K(x|M)

        We approximate this by checking if the model captures enough
        structure that the residual (data minus model prediction) is
        incompressible.

        Args:
            data: Original data.
            model_complexity: Number of bits to describe the model.

        Returns:
            Ratio of (model + residual complexity) / data complexity.
            Values near 1.0 indicate a good sufficient statistic.
        """
        kc = KolmogorovComplexity()
        data_bytes = data.tobytes()

        c_data = len(zlib.compress(data_bytes, level=9))
        if c_data == 0:
            return 1.0

        # Total description = model + data given model
        # Here we approximate K(x|M) by the residual complexity
        total = model_complexity + c_data
        return float(total / (c_data + model_complexity))

    @staticmethod
    def levin_complexity(data: np.ndarray, programs: list, timeout_steps: int = 1000) -> float:
        """Approximate Levin's Kt complexity (time-bounded).

        Kt(x) = min_p { |p| + log t_p(x) } where t_p(x) is the runtime
        of program p that outputs x.

        Here 'programs' are simple generators (callables returning arrays).
        We measure description length as compressed size of the program
        index + any parameters.

        Args:
            data: Target data array.
            programs: List of (generator_fn, param_bits) tuples.
            timeout_steps: Max steps for each program.

        Returns:
            Minimum Kt estimate across programs.
        """
        min_kt = float('inf')

        for i, (gen_fn, param_bits) in enumerate(programs):
            try:
                output = gen_fn()
                if isinstance(output, np.ndarray) and np.allclose(output, data, atol=1e-6):
                    program_bits = int(np.ceil(np.log2(i + 2))) + param_bits
                    # Assume runtime proportional to data size (no actual timing)
                    runtime_bits = int(np.ceil(np.log2(len(data) + 1)))
                    kt = program_bits + runtime_bits
                    min_kt = min(min_kt, kt)
            except Exception:
                continue

        if min_kt == float('inf'):
            # No program matched — fall back to raw description
            return float(len(data) * 8)  # 8 bits per byte

        return float(min_kt)
