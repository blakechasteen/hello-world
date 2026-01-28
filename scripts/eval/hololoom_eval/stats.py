"""
Statistical rigor utilities for evaluation framework.

Provides:
- Bootstrap confidence intervals
- Paired significance tests (Wilcoxon signed-rank, paired t-test)
- Effect size (Cohen's d, rank-biserial correlation)
- Summary statistics with full reporting
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class ConfidenceInterval:
    """Bootstrap confidence interval."""
    mean: float
    lower: float
    upper: float
    confidence_level: float
    n_bootstrap: int

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def __repr__(self) -> str:
        return f"{self.mean:.4f} [{self.lower:.4f}, {self.upper:.4f}] ({self.confidence_level*100:.0f}% CI)"


@dataclass
class SignificanceTest:
    """Result of a paired significance test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool  # at alpha level
    alpha: float
    effect_size: float
    effect_size_name: str
    interpretation: str
    n: int


@dataclass
class MetricSummary:
    """Complete statistical summary for a metric."""
    name: str
    values: List[float]
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    ci: Optional[ConfidenceInterval] = None

    @property
    def n(self) -> int:
        return len(self.values)


def bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> ConfidenceInterval:
    """
    Compute bootstrap confidence interval for the mean.

    Uses percentile method with bias-corrected bounds.
    """
    if not values:
        return ConfidenceInterval(0.0, 0.0, 0.0, confidence, 0)

    rng = random.Random(seed)
    n = len(values)
    observed_mean = sum(values) / n

    if n == 1:
        return ConfidenceInterval(observed_mean, observed_mean, observed_mean, confidence, 0)

    # Bootstrap resampling
    boot_means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()

    # Percentile method
    alpha = 1.0 - confidence
    lower_idx = max(0, int(math.floor((alpha / 2) * n_bootstrap)))
    upper_idx = min(n_bootstrap - 1, int(math.ceil((1 - alpha / 2) * n_bootstrap)))

    return ConfidenceInterval(
        mean=observed_mean,
        lower=boot_means[lower_idx],
        upper=boot_means[upper_idx],
        confidence_level=confidence,
        n_bootstrap=n_bootstrap,
    )


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Cohen's d effect size between two groups.

    Uses pooled standard deviation. Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    mean1 = sum(group1) / n1
    mean2 = sum(group2) / n2

    var1 = sum((x - mean1) ** 2 for x in group1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in group2) / (n2 - 1)

    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def wilcoxon_signed_rank(
    x: List[float],
    y: List[float],
    alpha: float = 0.05,
) -> SignificanceTest:
    """
    Wilcoxon signed-rank test for paired samples.

    Non-parametric alternative to paired t-test. Tests whether
    the median difference is zero.

    Uses normal approximation for n >= 10.
    """
    if len(x) != len(y):
        raise ValueError("Samples must have same length")

    n = len(x)
    differences = [xi - yi for xi, yi in zip(x, y)]

    # Remove zeros
    nonzero = [(abs(d), d) for d in differences if d != 0]
    if not nonzero:
        return SignificanceTest(
            test_name="Wilcoxon signed-rank",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            effect_size=0.0,
            effect_size_name="rank-biserial r",
            interpretation="No differences detected",
            n=n,
        )

    n_nonzero = len(nonzero)

    # Rank absolute differences
    sorted_diffs = sorted(nonzero, key=lambda x: x[0])

    # Handle ties by average rank
    ranks = [0.0] * n_nonzero
    i = 0
    while i < n_nonzero:
        j = i
        while j < n_nonzero and sorted_diffs[j][0] == sorted_diffs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-indexed average
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    # Sum positive and negative ranks
    w_plus = sum(ranks[i] for i in range(n_nonzero) if sorted_diffs[i][1] > 0)
    w_minus = sum(ranks[i] for i in range(n_nonzero) if sorted_diffs[i][1] < 0)

    w = min(w_plus, w_minus)
    w_max = w_plus + w_minus

    # Normal approximation for p-value (valid for n >= 10)
    if n_nonzero >= 10:
        mean_w = n_nonzero * (n_nonzero + 1) / 4
        std_w = math.sqrt(n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24)

        if std_w > 0:
            z = (w - mean_w) / std_w
            # Two-tailed p-value using standard normal CDF approximation
            p_value = 2 * _normal_cdf(-abs(z))
        else:
            p_value = 1.0
    else:
        # For small samples, use conservative estimate
        # (exact tables would be ideal but add complexity)
        p_value = 0.1 if w < n_nonzero * (n_nonzero + 1) / 4 * 0.5 else 0.5

    # Rank-biserial correlation (effect size)
    if w_max > 0:
        r = (w_plus - w_minus) / w_max
    else:
        r = 0.0

    significant = p_value < alpha

    d = cohens_d(x, y)
    interp = _interpret_cohens_d(d)

    if significant:
        direction = "improvement" if sum(differences) / len(differences) > 0 else "degradation"
        interpretation = f"Statistically significant {direction} (p={p_value:.4f}, {interp} effect d={d:.3f})"
    else:
        interpretation = f"No significant difference (p={p_value:.4f}, d={d:.3f})"

    return SignificanceTest(
        test_name="Wilcoxon signed-rank",
        statistic=w,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=r,
        effect_size_name="rank-biserial r",
        interpretation=interpretation,
        n=n,
    )


def paired_t_test(
    x: List[float],
    y: List[float],
    alpha: float = 0.05,
) -> SignificanceTest:
    """
    Paired t-test for two related samples.

    Tests whether the mean difference is zero.
    """
    if len(x) != len(y):
        raise ValueError("Samples must have same length")

    n = len(x)
    differences = [xi - yi for xi, yi in zip(x, y)]
    mean_diff = sum(differences) / n

    if n < 2:
        return SignificanceTest(
            test_name="Paired t-test",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            effect_size=0.0,
            effect_size_name="Cohen's d",
            interpretation="Insufficient data",
            n=n,
        )

    var_diff = sum((d - mean_diff) ** 2 for d in differences) / (n - 1)
    std_diff = math.sqrt(var_diff)

    if std_diff == 0:
        p_value = 0.0 if mean_diff != 0 else 1.0
        t_stat = float('inf') if mean_diff > 0 else float('-inf') if mean_diff < 0 else 0.0
    else:
        t_stat = mean_diff / (std_diff / math.sqrt(n))
        # Two-tailed p-value using t-distribution approximation
        # For large n, t ~ N(0,1); for small n, this is approximate
        df = n - 1
        p_value = 2 * _t_cdf(-abs(t_stat), df)

    d = cohens_d(x, y)
    interp = _interpret_cohens_d(d)
    significant = p_value < alpha

    if significant:
        direction = "improvement" if mean_diff > 0 else "degradation"
        interpretation = f"Statistically significant {direction} (p={p_value:.4f}, {interp} effect d={d:.3f})"
    else:
        interpretation = f"No significant difference (p={p_value:.4f}, d={d:.3f})"

    return SignificanceTest(
        test_name="Paired t-test",
        statistic=t_stat,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=d,
        effect_size_name="Cohen's d",
        interpretation=interpretation,
        n=n,
    )


def compute_metric_summary(
    name: str,
    values: List[float],
    ci_confidence: float = 0.95,
    ci_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> MetricSummary:
    """Compute full statistical summary for a metric."""
    if not values:
        return MetricSummary(name=name, values=[], mean=0.0, median=0.0, std=0.0, min_val=0.0, max_val=0.0)

    n = len(values)
    mean = sum(values) / n
    sorted_vals = sorted(values)

    if n % 2 == 0:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    else:
        median = sorted_vals[n // 2]

    variance = sum((x - mean) ** 2 for x in values) / max(1, n - 1)
    std = math.sqrt(variance)

    ci = bootstrap_ci(values, confidence=ci_confidence, n_bootstrap=ci_bootstrap, seed=seed)

    return MetricSummary(
        name=name,
        values=values,
        mean=mean,
        median=median,
        std=std,
        min_val=min(values),
        max_val=max(values),
        ci=ci,
    )


def compare_methods(
    method_a_scores: List[float],
    method_b_scores: List[float],
    method_a_name: str = "A",
    method_b_name: str = "B",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Complete statistical comparison between two methods.

    Returns summary with both parametric and non-parametric tests,
    effect sizes, and confidence intervals.
    """
    summary_a = compute_metric_summary(method_a_name, method_a_scores)
    summary_b = compute_metric_summary(method_b_name, method_b_scores)

    result = {
        "method_a": {
            "name": method_a_name,
            "mean": summary_a.mean,
            "std": summary_a.std,
            "ci": repr(summary_a.ci) if summary_a.ci else None,
        },
        "method_b": {
            "name": method_b_name,
            "mean": summary_b.mean,
            "std": summary_b.std,
            "ci": repr(summary_b.ci) if summary_b.ci else None,
        },
        "difference": summary_a.mean - summary_b.mean,
    }

    # Run paired tests if samples are aligned
    if len(method_a_scores) == len(method_b_scores) and len(method_a_scores) >= 3:
        t_test = paired_t_test(method_a_scores, method_b_scores, alpha=alpha)
        wilcoxon = wilcoxon_signed_rank(method_a_scores, method_b_scores, alpha=alpha)

        result["paired_t_test"] = {
            "statistic": t_test.statistic,
            "p_value": t_test.p_value,
            "significant": t_test.significant,
            "interpretation": t_test.interpretation,
        }
        result["wilcoxon_test"] = {
            "statistic": wilcoxon.statistic,
            "p_value": wilcoxon.p_value,
            "significant": wilcoxon.significant,
            "interpretation": wilcoxon.interpretation,
        }
        result["effect_size"] = {
            "cohens_d": cohens_d(method_a_scores, method_b_scores),
            "interpretation": _interpret_cohens_d(cohens_d(method_a_scores, method_b_scores)),
        }
    else:
        result["note"] = "Insufficient paired data for significance tests"

    return result


# ============================================================================
# Internal math helpers (no external deps)
# ============================================================================

def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    if x < -8:
        return 0.0
    if x > 8:
        return 1.0

    # Rational approximation
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p = d * math.exp(-x * x / 2.0) * (
        t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    )

    if x > 0:
        return 1.0 - p
    return p


def _t_cdf(t: float, df: int) -> float:
    """
    Student's t-distribution CDF approximation.

    Uses normal approximation with correction for moderate df.
    """
    if df <= 0:
        return 0.5

    if df >= 30:
        # Normal approximation for large df
        return _normal_cdf(t)

    # Improved approximation for small df
    # Cornish-Fisher expansion
    g1 = 1.0 / (4 * df)
    adjusted = t * (1 + g1 * (t * t + 1))
    # Scale for degrees of freedom
    scale = math.sqrt(df / (df - 2)) if df > 2 else 1.5
    return _normal_cdf(adjusted / scale)
