"""
Streaming Data Structures — CM5.4
==================================

Sublinear-space algorithms for processing data streams. Each structure
uses O(polylog(n)) space to answer queries about n items, trading
exactness for dramatic space savings.

Classes:
    CountMinSketch: Approximate frequency estimation
    HyperLogLog: Cardinality (distinct count) estimation
    ReservoirSampling: Uniform sample from a stream
    ExponentialHistogram: Sliding window bit counting

Key Concepts:
    - Count-Min Sketch: frequency upper bound with additive error epsilon * ||a||_1
    - HyperLogLog: cardinality estimate with ~1.04/sqrt(m) relative error
    - Reservoir sampling: each of n items has k/n probability of being in sample
    - Exponential histogram: (1+epsilon)-approximate sliding window counts

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import hashlib
import logging
import struct
from collections.abc import Hashable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CountMinSketch:
    """
    Count-Min Sketch for approximate frequency queries.

    Uses depth hash functions mapping items to a width x depth table.
    Point query returns the minimum across all hash positions,
    giving an upper bound on the true frequency.

    Error guarantee: P(estimate - true > epsilon * N) < delta
    with width = ceil(e/epsilon), depth = ceil(ln(1/delta)).

    Space: O(width * depth) = O((1/epsilon) * log(1/delta))
    """

    def __init__(self, width: int = 1000, depth: int = 5, seed: int = 42):
        """
        Args:
            width: Number of columns (controls accuracy)
            depth: Number of rows/hash functions (controls confidence)
            seed: Random seed for hash functions
        """
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int64)
        self._total = 0

        # Generate hash seeds
        rng = np.random.RandomState(seed)
        self._hash_a = rng.randint(1, 2**31, size=depth)
        self._hash_b = rng.randint(0, 2**31, size=depth)
        self._prime = 2**31 - 1

    def _hash(self, item: Hashable, i: int) -> int:
        """Hash item to column index for row i."""
        h = hash(item)
        return ((self._hash_a[i] * h + self._hash_b[i]) % self._prime) % self.width

    def add(self, item: Hashable, count: int = 1) -> None:
        """Add item to the sketch (increment its count)."""
        self._total += count
        for i in range(self.depth):
            j = self._hash(item, i)
            self.table[i, j] += count

    def query(self, item: Hashable) -> int:
        """
        Query approximate frequency of item.

        Returns the minimum count across all hash positions,
        which is an upper bound on the true frequency.
        """
        estimates = [self.table[i, self._hash(item, i)] for i in range(self.depth)]
        return min(estimates)

    @property
    def total_count(self) -> int:
        """Total number of items added."""
        return self._total

    @staticmethod
    def optimal_params(epsilon: float, delta: float) -> tuple[int, int]:
        """
        Compute optimal width and depth for desired error bounds.

        Args:
            epsilon: Additive error as fraction of total count
            delta: Failure probability

        Returns:
            (width, depth) parameters
        """
        width = int(np.ceil(np.e / epsilon))
        depth = int(np.ceil(np.log(1.0 / delta)))
        return width, depth


class HyperLogLog:
    """
    HyperLogLog for cardinality (distinct count) estimation.

    Uses the observation that for uniform hashes, the maximum number
    of leading zeros in any hash ~ log2(n_distinct).

    Uses 2^p registers (p bits for bucket selection), each storing
    the maximum leading-zero run. Final estimate uses harmonic mean
    with bias corrections.

    Space: O(2^p) bytes. Standard error: ~1.04 / sqrt(2^p).

    Args:
        p: Precision parameter (4-16). Uses 2^p = m registers.
           p=14 gives m=16384 registers, ~0.81% standard error.
    """

    def __init__(self, p: int = 14):
        if not 4 <= p <= 16:
            raise ValueError(f"p must be in [4, 16], got {p}")
        self.p = p
        self.m = 1 << p  # Number of registers
        self.registers = np.zeros(self.m, dtype=np.uint8)
        self._alpha = self._compute_alpha(self.m)

    @staticmethod
    def _compute_alpha(m: int) -> float:
        """Bias correction constant."""
        if m == 16:
            return 0.673
        elif m == 32:
            return 0.697
        elif m == 64:
            return 0.709
        else:
            return 0.7213 / (1 + 1.079 / m)

    def _hash_item(self, item: Hashable) -> int:
        """Hash item to 64-bit integer."""
        h = hashlib.sha256(str(item).encode()).digest()
        return struct.unpack('<Q', h[:8])[0]

    def _leading_zeros(self, value: int, max_bits: int = 64) -> int:
        """Count leading zeros after bucket bits."""
        if value == 0:
            return max_bits
        count = 0
        for bit in range(max_bits - 1, -1, -1):
            if value & (1 << bit):
                break
            count += 1
        return count

    def add(self, item: Hashable) -> None:
        """Add an item to the HLL."""
        h = self._hash_item(item)

        # First p bits determine the register
        j = h & (self.m - 1)

        # Remaining bits: count leading zeros + 1
        w = h >> self.p
        rho = self._leading_zeros(w, 64 - self.p) + 1

        self.registers[j] = max(self.registers[j], rho)

    def count(self) -> int:
        """
        Estimate the number of distinct items.

        Uses harmonic mean with linear counting correction for small
        cardinalities and large range correction for near-capacity.
        """
        # Raw HLL estimate
        indicator = np.sum(2.0 ** (-self.registers.astype(float)))
        estimate = self._alpha * self.m * self.m / indicator

        # Small range correction (linear counting)
        if estimate <= 2.5 * self.m:
            n_zeros = np.sum(self.registers == 0)
            if n_zeros > 0:
                estimate = self.m * np.log(self.m / n_zeros)

        # Large range correction
        two_32 = 2 ** 32
        if estimate > two_32 / 30:
            estimate = -two_32 * np.log(1 - estimate / two_32)

        return int(round(estimate))

    def merge(self, other: 'HyperLogLog') -> 'HyperLogLog':
        """Merge two HLL sketches (union operation)."""
        if self.p != other.p:
            raise ValueError("Cannot merge HLLs with different p")
        result = HyperLogLog(self.p)
        result.registers = np.maximum(self.registers, other.registers)
        return result


class ReservoirSampling:
    """
    Reservoir sampling: maintain a uniform random sample from a stream.

    Algorithm R (Vitter 1985): maintain k items such that at any point,
    each of the n items seen so far has exactly k/n probability of
    being in the reservoir.

    Space: O(k) regardless of stream length.
    """

    def __init__(self, k: int, seed: int | None = None):
        """
        Args:
            k: Reservoir size (number of items to keep)
            seed: Random seed
        """
        self.k = k
        self.reservoir: list[Any] = []
        self.n_seen = 0
        self._rng = np.random.RandomState(seed)

    def add(self, item: Any) -> None:
        """
        Process next item from the stream.

        First k items are added directly. After that, item i replaces
        a random reservoir element with probability k/i.
        """
        self.n_seen += 1

        if self.n_seen <= self.k:
            self.reservoir.append(item)
        else:
            j = self._rng.randint(0, self.n_seen)
            if j < self.k:
                self.reservoir[j] = item

    def sample(self) -> list[Any]:
        """Return the current reservoir (uniform sample)."""
        return list(self.reservoir)

    @property
    def is_full(self) -> bool:
        """Whether the reservoir has reached capacity."""
        return self.n_seen >= self.k

    def add_batch(self, items: list[Any]) -> None:
        """Add multiple items from the stream."""
        for item in items:
            self.add(item)

    @staticmethod
    def weighted_reservoir(
        items: list[Any],
        weights: np.ndarray,
        k: int,
        seed: int | None = None,
    ) -> list[Any]:
        """
        Weighted reservoir sampling (Efraimidis-Spirakis, 2006).

        Each item i has probability proportional to w_i of being selected.

        Args:
            items: Full list of items
            weights: Non-negative weights for each item
            k: Number of items to select
            seed: Random seed

        Returns:
            Weighted random sample of size k
        """
        rng = np.random.RandomState(seed)
        n = len(items)
        if k >= n:
            return list(items)

        # Key for each item: u^{1/w} where u ~ Uniform(0,1)
        u = rng.random(n)
        keys = u ** (1.0 / (weights + 1e-10))

        # Top-k keys
        top_k_indices = np.argpartition(keys, -k)[-k:]
        return [items[i] for i in top_k_indices]


class ExponentialHistogram:
    """
    Exponential histogram for sliding window counting.

    Maintains (1+epsilon)-approximate count of 1-bits in the
    last W positions of a binary stream, using O((1/epsilon) log^2 W) space.

    Idea: maintain "buckets" of exponentially increasing sizes.
    Each bucket records a timestamp and a count of 1-bits.
    At most (1/epsilon + 1) buckets of each size are kept.
    """

    def __init__(self, epsilon: float = 0.5, window_size: int = 1000):
        """
        Args:
            epsilon: Relative error bound
            window_size: Sliding window size W
        """
        self.epsilon = epsilon
        self.window_size = window_size
        self.max_buckets_per_size = int(np.ceil(1.0 / epsilon)) + 1

        # Buckets: list of (timestamp, size) pairs, newest first
        self._buckets: list[tuple[int, int]] = []
        self._timestamp = 0
        self._total = 0

    def add(self, bit: int) -> None:
        """
        Process next bit from the stream.

        Args:
            bit: 0 or 1
        """
        self._timestamp += 1

        # Expire old buckets outside the window
        cutoff = self._timestamp - self.window_size
        self._buckets = [(t, s) for t, s in self._buckets if t > cutoff]

        if bit == 1:
            # Add a new bucket of size 1
            self._buckets.insert(0, (self._timestamp, 1))
            self._total += 1

            # Merge if too many buckets of the same size
            self._merge_buckets()

    def _merge_buckets(self) -> None:
        """Merge excess buckets of the same size."""
        # Count buckets by size
        i = 0
        while i < len(self._buckets):
            size = self._buckets[i][1]
            # Find all buckets of this size
            same_size = []
            j = i
            while j < len(self._buckets) and self._buckets[j][1] == size:
                same_size.append(j)
                j += 1

            if len(same_size) > self.max_buckets_per_size:
                # Merge the two oldest (rightmost) of this size
                idx1 = same_size[-2]
                idx2 = same_size[-1]
                merged_time = self._buckets[idx1][0]  # Keep newer timestamp
                merged_size = size * 2
                self._buckets[idx1] = (merged_time, merged_size)
                del self._buckets[idx2]
                # Don't advance i — recheck this size
            else:
                i = j

    def count(self) -> int:
        """
        Estimate the count of 1-bits in the current window.

        The estimate is the sum of all full buckets plus half
        of the oldest (possibly partial) bucket.
        """
        if not self._buckets:
            return 0

        cutoff = self._timestamp - self.window_size

        total = 0
        oldest_size = 0
        for t, s in self._buckets:
            if t > cutoff:
                total += s
                oldest_size = s

        # Subtract half of the oldest bucket (may be partially outside window)
        return max(0, total - oldest_size // 2)

    @property
    def exact_space(self) -> int:
        """Number of buckets currently maintained."""
        return len(self._buckets)
