"""
Nonparametric Tests — Distribution-Free Hypothesis Testing
===========================================================

Statistical tests that make no assumptions about the underlying
distribution. Rank-based methods for comparing groups.

Core Concepts:
- Wilcoxon Signed-Rank: Paired test (replaces paired t-test)
- Mann-Whitney U: Unpaired two-sample test (replaces independent t-test)
- Kruskal-Wallis: Multi-group test (replaces one-way ANOVA)
- Friedman: Repeated measures (replaces repeated-measures ANOVA)
- Multiple Testing Correction: Bonferroni, Holm, BH

Mathematical Foundation:
Wilcoxon: W = Σ sign(dᵢ) R(|dᵢ|), normal approx for large n
Mann-Whitney: U = Σᵢ Σⱼ I(Xᵢ > Yⱼ)
Kruskal-Wallis: H = 12/(N(N+1)) Σ nⱼ(R̄ⱼ - R̄)² ~ χ²(k-1)
Friedman: Q = 12/(bk(k+1)) Σ R̄ⱼ² - 3b(k+1) ~ χ²(k-1)

Applications to Warp Space:
- Comparing scoring term distributions across modes
- Testing whether model routing choices differ significantly
- Non-Gaussian hypothesis testing for embedding metrics

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging

import numpy as np

try:
    from scipy.stats import chi2 as scipy_chi2
    from scipy.stats import norm as scipy_norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


# ============================================================================
# NORMAL APPROXIMATION FALLBACK
# ============================================================================

def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    if HAS_SCIPY:
        return float(scipy_norm.cdf(x))
    # Hart approximation
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.2316419 * x)
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p = d * np.exp(-x * x / 2.0) * (
        t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
        t * (-1.821255978 + t * 1.330274429))))
    )
    return 0.5 + sign * (0.5 - p)


def _chi2_cdf(x: float, df: int) -> float:
    """Chi-squared CDF approximation (Wilson-Hilferty)."""
    if HAS_SCIPY:
        return float(scipy_chi2.cdf(x, df))
    if df <= 0 or x <= 0:
        return 0.0
    # Wilson-Hilferty approximation
    z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / np.sqrt(2 / (9 * df))
    return _normal_cdf(z)


# ============================================================================
# WILCOXON SIGNED-RANK TEST
# ============================================================================

class WilcoxonTest:
    """Wilcoxon signed-rank test for paired samples.

    Nonparametric alternative to the paired t-test.

    Example:
        >>> wt = WilcoxonTest()
        >>> x = np.array([125, 115, 130, 140, 140, 115, 140, 125, 140, 135])
        >>> y = np.array([110, 122, 125, 120, 140, 124, 123, 137, 135, 145])
        >>> stat, p = wt.signed_rank(x, y)
    """

    @staticmethod
    def signed_rank(
        x: np.ndarray,
        y: np.ndarray,
        alternative: str = "two-sided",
    ) -> tuple[float, float]:
        """Wilcoxon signed-rank test.

        Args:
            x, y: Paired samples of equal length
            alternative: "two-sided", "greater", or "less"

        Returns:
            (test_statistic W, p_value)
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        d = x - y
        d = d[d != 0]  # Remove zero differences
        n = len(d)

        if n == 0:
            return 0.0, 1.0

        ranks = np.argsort(np.argsort(np.abs(d))) + 1.0

        W_plus = np.sum(ranks[d > 0])
        W_minus = np.sum(ranks[d < 0])

        if alternative == "greater":
            W = W_plus
        elif alternative == "less":
            W = W_minus
        else:
            W = min(W_plus, W_minus)

        # Normal approximation for large n
        mu = n * (n + 1) / 4
        sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

        if sigma < 1e-10:
            return float(W), 1.0

        z = (W - mu) / sigma

        if alternative == "two-sided":
            p = 2 * (1 - _normal_cdf(abs(z)))
        elif alternative == "greater":
            p = 1 - _normal_cdf(z)
        else:
            p = _normal_cdf(z)

        return float(W), float(max(p, 0.0))


# ============================================================================
# MANN-WHITNEY U TEST
# ============================================================================

class MannWhitneyU:
    """Mann-Whitney U test for two independent samples.

    Nonparametric alternative to the independent-samples t-test.

    Example:
        >>> mwu = MannWhitneyU()
        >>> x = np.array([19, 22, 16, 29, 24])
        >>> y = np.array([20, 11, 17, 12, 15])
        >>> stat, p = mwu.test(x, y)
    """

    @staticmethod
    def test(
        x: np.ndarray,
        y: np.ndarray,
        alternative: str = "two-sided",
    ) -> tuple[float, float]:
        """Mann-Whitney U test.

        Args:
            x, y: Two independent samples
            alternative: "two-sided", "greater", or "less"

        Returns:
            (U_statistic, p_value)
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)

        # Compute U directly
        U = 0.0
        for xi in x:
            U += np.sum(xi > y) + 0.5 * np.sum(xi == y)

        # Normal approximation
        mu = nx * ny / 2
        sigma = np.sqrt(nx * ny * (nx + ny + 1) / 12)

        if sigma < 1e-10:
            return float(U), 1.0

        z = (U - mu) / sigma

        if alternative == "two-sided":
            p = 2 * (1 - _normal_cdf(abs(z)))
        elif alternative == "greater":
            p = 1 - _normal_cdf(z)
        else:
            p = _normal_cdf(z)

        return float(U), float(max(p, 0.0))


# ============================================================================
# KRUSKAL-WALLIS TEST
# ============================================================================

class KruskalWallis:
    """Kruskal-Wallis H test for multiple independent groups.

    Nonparametric alternative to one-way ANOVA.

    Example:
        >>> kw = KruskalWallis()
        >>> g1 = np.array([7, 14, 14, 13, 12, 9, 6, 14, 12, 8])
        >>> g2 = np.array([15, 17, 13, 15, 15, 13, 9, 12, 10, 8])
        >>> g3 = np.array([6, 8, 8, 9, 5, 14, 13, 8, 10, 9])
        >>> stat, p = kw.test(g1, g2, g3)
    """

    @staticmethod
    def test(*groups: np.ndarray) -> tuple[float, float]:
        """Kruskal-Wallis H test.

        Args:
            *groups: Two or more sample arrays

        Returns:
            (H_statistic, p_value) under chi-squared(k-1)
        """
        k = len(groups)
        if k < 2:
            raise ValueError("Need at least 2 groups")

        groups = [np.asarray(g, dtype=float) for g in groups]
        all_data = np.concatenate(groups)
        N = len(all_data)

        # Rank all observations together
        order = np.argsort(all_data)
        ranks = np.empty(N)
        ranks[order] = np.arange(1, N + 1, dtype=float)

        # Handle ties: average ranks
        unique_vals, counts = np.unique(all_data, return_counts=True)
        for val, count in zip(unique_vals, counts):
            if count > 1:
                mask = all_data == val
                ranks[mask] = np.mean(ranks[mask])

        # Compute H statistic
        H = 0.0
        start = 0
        for g in groups:
            n_i = len(g)
            group_ranks = ranks[start:start + n_i]
            R_bar_i = np.mean(group_ranks)
            H += n_i * (R_bar_i - (N + 1) / 2) ** 2
            start += n_i

        H = 12 * H / (N * (N + 1))

        # Tie correction
        tie_sum = np.sum(counts ** 3 - counts)
        if tie_sum > 0:
            H = H / (1 - tie_sum / (N ** 3 - N))

        # P-value from chi-squared(k-1)
        p = 1 - _chi2_cdf(H, k - 1)

        return float(H), float(max(p, 0.0))


# ============================================================================
# FRIEDMAN TEST
# ============================================================================

class FriedmanTest:
    """Friedman test for repeated measures.

    Nonparametric alternative to repeated-measures ANOVA.

    Example:
        >>> ft = FriedmanTest()
        >>> # 5 subjects, 3 treatments each
        >>> t1 = np.array([4, 6, 3, 4, 3])
        >>> t2 = np.array([5, 8, 4, 5, 4])
        >>> t3 = np.array([7, 9, 6, 7, 5])
        >>> stat, p = ft.test(t1, t2, t3)
    """

    @staticmethod
    def test(*groups: np.ndarray) -> tuple[float, float]:
        """Friedman test.

        Args:
            *groups: k treatment arrays, each of length n (blocks)

        Returns:
            (Q_statistic, p_value) under chi-squared(k-1)
        """
        k = len(groups)
        if k < 2:
            raise ValueError("Need at least 2 groups")

        groups = [np.asarray(g, dtype=float) for g in groups]
        n = len(groups[0])
        for g in groups:
            if len(g) != n:
                raise ValueError("All groups must have the same length (balanced design)")

        # Rank within each block (row)
        data = np.column_stack(groups)  # (n, k)
        ranked = np.zeros_like(data)
        for i in range(n):
            order = np.argsort(data[i])
            r = np.empty(k)
            r[order] = np.arange(1, k + 1, dtype=float)

            # Handle ties
            unique_vals, counts = np.unique(data[i], return_counts=True)
            for val, count in zip(unique_vals, counts):
                if count > 1:
                    mask = data[i] == val
                    r[mask] = np.mean(r[mask])

            ranked[i] = r

        # Friedman Q statistic
        R_bar = np.mean(ranked, axis=0)  # Mean rank per treatment
        Q = 12 * n / (k * (k + 1)) * np.sum((R_bar - (k + 1) / 2) ** 2)

        p = 1 - _chi2_cdf(Q, k - 1)

        return float(Q), float(max(p, 0.0))


# ============================================================================
# MULTIPLE TESTING CORRECTION
# ============================================================================

def multiple_testing_correction(
    p_values: np.ndarray,
    method: str = "bonferroni",
) -> np.ndarray:
    """Correct p-values for multiple comparisons.

    Args:
        p_values: Raw p-values (m,)
        method: "bonferroni", "holm", or "benjamini-hochberg" (BH/FDR)

    Returns:
        Adjusted p-values (m,), clipped to [0, 1]
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)

    if method == "bonferroni":
        adjusted = p_values * m

    elif method == "holm":
        # Holm step-down
        order = np.argsort(p_values)
        adjusted = np.empty(m)
        running_max = 0.0
        for i, idx in enumerate(order):
            val = p_values[idx] * (m - i)
            running_max = max(running_max, val)
            adjusted[idx] = running_max

    elif method == "benjamini-hochberg" or method == "bh" or method == "fdr":
        # Benjamini-Hochberg FDR
        order = np.argsort(p_values)
        adjusted = np.empty(m)
        running_min = 1.0
        for i in range(m - 1, -1, -1):
            idx = order[i]
            val = p_values[idx] * m / (i + 1)
            running_min = min(running_min, val)
            adjusted[idx] = running_min

    else:
        raise ValueError(f"Unknown method: {method}. Use 'bonferroni', 'holm', or 'benjamini-hochberg'")

    return np.clip(adjusted, 0.0, 1.0)
