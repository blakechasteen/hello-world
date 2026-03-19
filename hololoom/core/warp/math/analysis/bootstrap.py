"""
Bootstrap and Resampling Methods — ST6.3-6.4
=============================================

Non-parametric inference via resampling. No distributional assumptions
needed — the data itself estimates the sampling distribution.

Classes:
    Bootstrap: Confidence intervals via percentile and BCa methods
    PermutationTest: Non-parametric hypothesis testing
    Jackknife: Leave-one-out bias and variance estimation

Mathematical Foundation:
    Bootstrap principle: F_hat -> X* mimics F -> X
    BCa correction: accounts for bias and skewness of bootstrap distribution
    Permutation test: exact p-value under null by exhaustive/random permutation
    Jackknife: bias = (n-1)(theta_dot - theta_hat), se from pseudovalues

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)


class Bootstrap:
    """
    Bootstrap confidence intervals for arbitrary statistics.

    Given data x_1, ..., x_n and a statistic T(x):
    1. Resample with replacement: x*_1, ..., x*_n
    2. Compute T(x*) for each resample
    3. Use the distribution of T(x*) for inference
    """

    @staticmethod
    def confidence_interval(
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
        n_resamples: int = 10000,
        alpha: float = 0.05,
        method: str = "percentile",
        seed: int | None = None,
    ) -> tuple[float, float]:
        """
        Compute a (1-alpha) confidence interval via bootstrap.

        Args:
            data: Sample data, shape (n,) or (n, p)
            statistic: Function computing the statistic of interest
            n_resamples: Number of bootstrap resamples
            alpha: Significance level (e.g., 0.05 for 95% CI)
            method: "percentile" or "bca" (bias-corrected accelerated)
            seed: Random seed

        Returns:
            (lower, upper) confidence bounds
        """
        rng = np.random.RandomState(seed)
        n = len(data)
        theta_hat = statistic(data)

        # Generate bootstrap distribution
        boot_stats = np.zeros(n_resamples)
        for b in range(n_resamples):
            indices = rng.randint(0, n, size=n)
            boot_sample = data[indices]
            boot_stats[b] = statistic(boot_sample)

        if method == "percentile":
            lower = np.percentile(boot_stats, 100 * alpha / 2)
            upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
            return lower, upper

        elif method == "bca":
            return Bootstrap._bca_interval(
                data, statistic, boot_stats, theta_hat, alpha
            )

        else:
            raise ValueError(f"Unknown method: {method}. Use 'percentile' or 'bca'.")

    @staticmethod
    def _bca_interval(
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
        boot_stats: np.ndarray,
        theta_hat: float,
        alpha: float,
    ) -> tuple[float, float]:
        """
        BCa (bias-corrected and accelerated) confidence interval.

        Corrects for:
        - Bias: median of bootstrap != theta_hat
        - Acceleration: skewness of the bootstrap distribution
        """
        from scipy.stats import norm as _norm
        n = len(data)

        # Bias correction: z_0 = Phi^{-1}(proportion of boot < theta_hat)
        prop_below = np.mean(boot_stats < theta_hat)
        prop_below = np.clip(prop_below, 1e-10, 1 - 1e-10)
        z_0 = _norm.ppf(prop_below)

        # Acceleration: estimated from jackknife influence values
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jack_sample = np.delete(data, i, axis=0)
            jackknife_stats[i] = statistic(jack_sample)

        theta_dot = jackknife_stats.mean()
        diff = theta_dot - jackknife_stats
        a_hat = np.sum(diff ** 3) / (6 * (np.sum(diff ** 2)) ** 1.5 + 1e-10)

        # Adjusted percentiles
        z_alpha_low = _norm.ppf(alpha / 2)
        z_alpha_high = _norm.ppf(1 - alpha / 2)

        alpha1 = _norm.cdf(z_0 + (z_0 + z_alpha_low) / (1 - a_hat * (z_0 + z_alpha_low)))
        alpha2 = _norm.cdf(z_0 + (z_0 + z_alpha_high) / (1 - a_hat * (z_0 + z_alpha_high)))

        lower = np.percentile(boot_stats, 100 * alpha1)
        upper = np.percentile(boot_stats, 100 * alpha2)

        return lower, upper

    @staticmethod
    def bootstrap_samples(
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
        n_resamples: int = 10000,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Generate the full bootstrap distribution of a statistic.

        Args:
            data: Sample data
            statistic: Function computing the statistic
            n_resamples: Number of resamples
            seed: Random seed

        Returns:
            Array of bootstrap statistics, shape (n_resamples,)
        """
        rng = np.random.RandomState(seed)
        n = len(data)
        stats = np.zeros(n_resamples)
        for b in range(n_resamples):
            indices = rng.randint(0, n, size=n)
            stats[b] = statistic(data[indices])
        return stats

    @staticmethod
    def standard_error(
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
        n_resamples: int = 10000,
        seed: int | None = None,
    ) -> float:
        """Bootstrap standard error of the statistic."""
        stats = Bootstrap.bootstrap_samples(data, statistic, n_resamples, seed)
        return float(np.std(stats, ddof=1))


class PermutationTest:
    """
    Permutation test for hypothesis testing.

    Under H_0 (no difference between groups), the group labels are
    exchangeable. The p-value is the fraction of permutations where
    the test statistic is at least as extreme as observed.

    Exact for small n; Monte Carlo approximation for large n.
    """

    @staticmethod
    def test(
        x: np.ndarray,
        y: np.ndarray,
        statistic: Callable[[np.ndarray, np.ndarray], float] | None = None,
        n_permutations: int = 10000,
        alternative: str = "two-sided",
        seed: int | None = None,
    ) -> tuple[float, float]:
        """
        Perform a two-sample permutation test.

        Args:
            x: First sample, shape (n_x,)
            y: Second sample, shape (n_y,)
            statistic: Test statistic T(x, y). Default: difference of means.
            n_permutations: Number of random permutations
            alternative: "two-sided", "greater", or "less"
            seed: Random seed

        Returns:
            (observed_statistic, p_value)
        """
        rng = np.random.RandomState(seed)

        if statistic is None:
            statistic = lambda a, b: np.mean(a) - np.mean(b)

        observed = statistic(x, y)
        combined = np.concatenate([x, y])
        n_x = len(x)
        n_total = len(combined)

        count = 0
        for _ in range(n_permutations):
            perm = rng.permutation(n_total)
            x_perm = combined[perm[:n_x]]
            y_perm = combined[perm[n_x:]]
            perm_stat = statistic(x_perm, y_perm)

            if alternative == "two-sided":
                if abs(perm_stat) >= abs(observed):
                    count += 1
            elif alternative == "greater":
                if perm_stat >= observed:
                    count += 1
            elif alternative == "less":
                if perm_stat <= observed:
                    count += 1

        p_value = (count + 1) / (n_permutations + 1)  # +1 for continuity
        return observed, p_value

    @staticmethod
    def paired_test(
        x: np.ndarray,
        y: np.ndarray,
        statistic: Callable[[np.ndarray], float] | None = None,
        n_permutations: int = 10000,
        seed: int | None = None,
    ) -> tuple[float, float]:
        """
        Paired permutation test.

        Under H_0, the sign of each difference is random.

        Args:
            x: First sample, shape (n,)
            y: Second sample, shape (n,) (paired with x)
            statistic: Test statistic on differences. Default: mean.
            n_permutations: Number of permutations
            seed: Random seed

        Returns:
            (observed_statistic, p_value)
        """
        rng = np.random.RandomState(seed)

        if statistic is None:
            statistic = np.mean

        diffs = x - y
        observed = statistic(diffs)
        n = len(diffs)

        count = 0
        for _ in range(n_permutations):
            signs = rng.choice([-1, 1], size=n)
            perm_stat = statistic(signs * diffs)
            if abs(perm_stat) >= abs(observed):
                count += 1

        p_value = (count + 1) / (n_permutations + 1)
        return observed, p_value


class Jackknife:
    """
    Jackknife resampling for bias and variance estimation.

    Leave-one-out: compute statistic on n subsets of size n-1.
    The pseudovalues theta_i = n*theta_hat - (n-1)*theta_{-i}
    estimate the influence of each observation.
    """

    @staticmethod
    def estimate(
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
    ) -> tuple[float, float]:
        """
        Jackknife estimate and standard error.

        Args:
            data: Sample data, shape (n,)
            statistic: Function computing the statistic

        Returns:
            (jackknife_estimate, standard_error)
        """
        n = len(data)
        theta_hat = statistic(data)

        # Leave-one-out statistics
        jack_stats = np.zeros(n)
        for i in range(n):
            jack_sample = np.delete(data, i, axis=0)
            jack_stats[i] = statistic(jack_sample)

        # Pseudovalues
        pseudovalues = n * theta_hat - (n - 1) * jack_stats

        # Jackknife estimate (bias-corrected)
        jack_estimate = pseudovalues.mean()

        # Standard error
        jack_se = np.sqrt(np.var(pseudovalues, ddof=1) / n)

        return jack_estimate, jack_se

    @staticmethod
    def bias(
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
    ) -> float:
        """
        Estimate bias of the statistic.

        bias = (n-1)(theta_dot - theta_hat)
        where theta_dot is the mean of leave-one-out estimates.

        Args:
            data: Sample data
            statistic: Function computing the statistic

        Returns:
            Estimated bias
        """
        n = len(data)
        theta_hat = statistic(data)

        theta_dot = 0.0
        for i in range(n):
            jack_sample = np.delete(data, i, axis=0)
            theta_dot += statistic(jack_sample)
        theta_dot /= n

        return (n - 1) * (theta_dot - theta_hat)

    @staticmethod
    def influence_values(
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
    ) -> np.ndarray:
        """
        Compute jackknife influence values (empirical influence function).

        influence_i = (n-1)(theta_hat - theta_{-i})

        Large influence = observation i strongly affects the statistic.

        Args:
            data: Sample data
            statistic: Function computing the statistic

        Returns:
            Influence values, shape (n,)
        """
        n = len(data)
        theta_hat = statistic(data)

        influences = np.zeros(n)
        for i in range(n):
            jack_sample = np.delete(data, i, axis=0)
            theta_i = statistic(jack_sample)
            influences[i] = (n - 1) * (theta_hat - theta_i)

        return influences
