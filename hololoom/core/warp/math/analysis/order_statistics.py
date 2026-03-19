"""
Order Statistics & L-Moments — Rank-Based Distributional Analysis
=================================================================

Distribution-free statistics based on sorted samples and their
linear combinations.

Core Concepts:
- Order statistics: k-th smallest value, range, midrange
- L-moments: Linear combinations of order statistics (robust alternatives to moments)
- Probability-weighted moments: Intermediate quantities for L-moment computation

Mathematical Foundation:
X_{(k:n)} = k-th order statistic of sample of size n
λ_r = (1/r) Σ_{k=0}^{r-1} (-1)^k C(r-1,k) E[X_{(r-k:r)}]    (L-moments)
β_r = E[X_{(1:1)} F(X)^r]                                       (PWMs)

Applications to Warp Space:
- Robust location/scale estimation for embedding distributions
- L-moment ratio diagrams for distribution identification
- Outlier-resistant feature statistics for memory scoring

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# ORDER STATISTICS
# ============================================================================

class OrderStatistics:
    """
    Classical order statistics: functionals of the sorted sample.

    Given a sample X_1, ..., X_n, the order statistics are
    X_{(1)} ≤ X_{(2)} ≤ ... ≤ X_{(n)}.
    """

    @staticmethod
    def kth(data: np.ndarray, k: int) -> float:
        """
        k-th order statistic X_{(k)}.

        Args:
            data: Sample array.
            k: Order (1-indexed, so k=1 is the minimum).

        Returns:
            X_{(k)}.
        """
        sorted_data = np.sort(data)
        if k < 1 or k > len(sorted_data):
            raise ValueError(f"k={k} out of range [1, {len(sorted_data)}]")
        return float(sorted_data[k - 1])

    @staticmethod
    def range_stat(data: np.ndarray) -> float:
        """
        Sample range: X_{(n)} - X_{(1)}.
        """
        return float(np.max(data) - np.min(data))

    @staticmethod
    def midrange(data: np.ndarray) -> float:
        """
        Midrange: (X_{(1)} + X_{(n)}) / 2.
        """
        return float((np.min(data) + np.max(data)) / 2.0)

    @staticmethod
    def interquartile_range(data: np.ndarray) -> float:
        """IQR: Q3 - Q1."""
        return float(np.percentile(data, 75) - np.percentile(data, 25))

    @staticmethod
    def trimmed_mean(data: np.ndarray, proportion: float = 0.1) -> float:
        """
        Trimmed mean: mean after removing proportion from each tail.

        Args:
            data: Sample array.
            proportion: Fraction to trim from each end (0 to 0.5).

        Returns:
            Trimmed mean.
        """
        sorted_data = np.sort(data)
        n = len(sorted_data)
        k = int(n * proportion)
        if k == 0:
            return float(np.mean(sorted_data))
        return float(np.mean(sorted_data[k:n - k]))

    @staticmethod
    def winsorized_mean(data: np.ndarray, proportion: float = 0.1) -> float:
        """
        Winsorized mean: replace tail values with boundary values, then average.
        """
        sorted_data = np.sort(data.copy())
        n = len(sorted_data)
        k = int(n * proportion)
        if k > 0:
            sorted_data[:k] = sorted_data[k]
            sorted_data[n - k:] = sorted_data[n - k - 1]
        return float(np.mean(sorted_data))


# ============================================================================
# PROBABILITY-WEIGHTED MOMENTS
# ============================================================================

class _PWMHelper:
    """Probability-weighted moments for L-moment computation."""

    @staticmethod
    def probability_weighted_moments(data: np.ndarray, order: int = 4) -> np.ndarray:
        """
        Compute probability-weighted moments β_0, β_1, ..., β_{order-1}.

        β_r = (1/n) Σ_{j=1}^{n} C(j-1, r) / C(n-1, r) · x_{(j)}

        where x_{(j)} are the sorted sample values.

        Args:
            data: Sample array.
            order: Number of PWMs to compute.

        Returns:
            Array of PWMs [β_0, β_1, ..., β_{order-1}].
        """
        x = np.sort(data)
        n = len(x)
        betas = np.zeros(order)

        for r in range(order):
            total = 0.0
            for j in range(n):
                # C(j, r) / C(n-1, r)
                if j < r:
                    continue
                comb_j = 1.0
                comb_n = 1.0
                for i in range(r):
                    comb_j *= (j - i) / (i + 1)
                    comb_n *= (n - 1 - i) / (i + 1)
                if comb_n > 0:
                    total += x[j] * comb_j / comb_n
            betas[r] = total / n

        return betas


# ============================================================================
# L-MOMENTS
# ============================================================================

class LMoments:
    """
    L-moments: robust analogs of conventional moments.

    L-moments are linear combinations of order statistics:
        λ_1 = L-mean (= sample mean)
        λ_2 = L-scale (measures dispersion)
        τ_3 = λ_3/λ_2 = L-skewness
        τ_4 = λ_4/λ_2 = L-kurtosis

    Advantages over conventional moments:
    - Always exist if the mean exists
    - Less sensitive to outliers
    - Better characterize distribution shape
    - L-moment ratio diagram identifies distribution families
    """

    @staticmethod
    def compute(data: np.ndarray, order: int = 4) -> np.ndarray:
        """
        Compute L-moments λ_1, λ_2, ..., λ_{order} from sample data.

        Uses PWM-based estimation:
            λ_1 = β_0
            λ_2 = 2β_1 - β_0
            λ_3 = 6β_2 - 6β_1 + β_0
            λ_4 = 20β_3 - 30β_2 + 12β_1 - β_0

        Args:
            data: Sample array (must have len >= order).
            order: Number of L-moments to compute (default 4).

        Returns:
            Array [λ_1, λ_2, ..., λ_{order}].
        """
        if len(data) < order:
            raise ValueError(f"Need at least {order} data points")

        betas = _PWMHelper.probability_weighted_moments(data, order)

        # Convert PWMs to L-moments using the shift matrix
        # λ_r = Σ_{k=0}^{r-1} (-1)^{r-1-k} C(r-1,k) C(r-1+k,k) β_k
        lmom = np.zeros(order)

        for r in range(1, order + 1):
            total = 0.0
            for k in range(r):
                sign = (-1) ** (r - 1 - k)
                # C(r-1, k) * C(r-1+k, k)
                c1 = 1.0
                c2 = 1.0
                for i in range(k):
                    c1 *= (r - 1 - i) / (i + 1)
                    c2 *= (r - 1 + k - i) / (i + 1)
                total += sign * c1 * c2 * betas[k]
            lmom[r - 1] = total

        return lmom

    @staticmethod
    def ratios(data: np.ndarray) -> dict:
        """
        Compute L-moment ratios for distribution identification.

        Returns:
            Dict with keys: l_mean, l_scale, l_skewness, l_kurtosis.
        """
        lmom = LMoments.compute(data, 4)

        result = {
            'l_mean': float(lmom[0]),
            'l_scale': float(lmom[1]),
            'l_skewness': float(lmom[2] / lmom[1]) if abs(lmom[1]) > 1e-14 else 0.0,
            'l_kurtosis': float(lmom[3] / lmom[1]) if abs(lmom[1]) > 1e-14 else 0.0,
        }

        return result

    @staticmethod
    def probability_weighted_moments(data: np.ndarray,
                                     order: int = 4) -> np.ndarray:
        """
        Public interface to PWM computation.

        Args:
            data: Sample array.
            order: Number of PWMs.

        Returns:
            Array of PWMs.
        """
        return _PWMHelper.probability_weighted_moments(data, order)
