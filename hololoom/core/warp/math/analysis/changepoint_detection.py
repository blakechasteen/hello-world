"""
Changepoint Detection — Regime Shifts in Time Series
=====================================================

Algorithms for detecting abrupt changes in the statistical properties
of a time series: mean shifts, variance changes, and trend breaks.

Core Concepts:
- CUSUM: Cumulative sum control chart for sequential detection
- PELT: Pruned Exact Linear Time — optimal segmentation via dynamic programming
- Bayesian Online Changepoint Detection: posterior over run lengths

Mathematical Foundation:
CUSUM:  S_t = max(0, S_{t-1} + (x_t - μ₀) - drift)
         Alarm when S_t > threshold

PELT (dynamic programming):
  F(t) = min_{s < t} [F(s) + C(y_{s+1:t}) + β]
  where C = segment cost, β = penalty (BIC: ½ k ln n)

Bayesian:
  P(r_t | x_{1:t}) ∝ P(x_t | r_t, x_{1:t}) P(r_t | r_{t-1})
  Run length r_t = time since last changepoint.
  P(r_t = 0) = Σ P(r_{t-1}) × prior_rate   (changepoint occurred)
  P(r_t = r_{t-1}+1) = P(r_{t-1}) × (1 - prior_rate)  (no change)

Applications to Warp Space:
- Detecting regime shifts in scoring trajectories
- Identifying convergence/divergence transitions
- Monitoring knowledge graph evolution for structural breaks
- Anomaly detection in embedding drift

Author: HoloLoom Team
Date: 2025-10-26
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.stats import norm as scipy_norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.debug("scipy not available; using numpy normal approximations")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Changepoint:
    """
    A detected changepoint in a time series.

    Attributes:
        index: Position in the series where the change occurs
        confidence: Confidence score in [0, 1]
        type: Kind of change: 'mean_shift', 'variance_shift', or 'trend_change'
    """
    index: int
    confidence: float
    type: str

    def __post_init__(self):
        valid_types = {'mean_shift', 'variance_shift', 'trend_change'}
        if self.type not in valid_types:
            raise ValueError(f"type must be one of {valid_types}, got '{self.type}'")


@dataclass
class ChangepointResult:
    """
    Result of changepoint detection.

    Attributes:
        changepoints: Detected changepoints, ordered by index
        segments: Data segments between changepoints
        bic: Bayesian Information Criterion for the segmented model
    """
    changepoints: list[Changepoint]
    segments: list[np.ndarray]
    bic: float = 0.0


# ============================================================================
# CUSUM
# ============================================================================

class CUSUM:
    """
    Cumulative Sum (CUSUM) control chart for sequential changepoint detection.

    Monitors deviations from a target mean. When the cumulative sum of
    centered observations exceeds a threshold, a changepoint is declared.

    Detects both upward and downward shifts (two-sided).
    """

    @staticmethod
    def detect(series: np.ndarray,
               threshold: float = 4.0,
               drift: float = 0.5,
               target: float | None = None) -> list[Changepoint]:
        """
        Detect changepoints via two-sided CUSUM.

        S⁺_t = max(0, S⁺_{t-1} + (x_t - μ) - k)   upward shift
        S⁻_t = max(0, S⁻_{t-1} - (x_t - μ) - k)   downward shift

        Alarm when S⁺_t > h or S⁻_t > h.

        Args:
            series: 1D time series
            threshold: Decision boundary h (higher = fewer false alarms)
            drift: Allowance parameter k (typically σ/2)
            target: Target mean μ₀ (estimated from first quarter if None)

        Returns:
            List of detected changepoints with confidence scores
        """
        series = np.asarray(series, dtype=float)
        n = len(series)

        if target is None:
            # Estimate from first quarter of data
            warmup = max(n // 4, 10)
            target = np.mean(series[:min(warmup, n)])

        std_est = np.std(series[:min(n // 4, max(10, n))], ddof=1)
        if std_est < 1e-15:
            std_est = 1.0

        # Normalize drift by estimated std
        k = drift * std_est
        h = threshold * std_est

        s_pos = 0.0
        s_neg = 0.0
        changepoints = []
        last_alarm = -1  # Prevent duplicate detections

        for t in range(n):
            deviation = series[t] - target
            s_pos = max(0.0, s_pos + deviation - k)
            s_neg = max(0.0, s_neg - deviation - k)

            triggered = False
            if s_pos > h:
                triggered = True
                confidence = min(1.0, s_pos / (2 * h))
                cp_type = 'mean_shift'
                s_pos = 0.0
            elif s_neg > h:
                triggered = True
                confidence = min(1.0, s_neg / (2 * h))
                cp_type = 'mean_shift'
                s_neg = 0.0

            if triggered and t > last_alarm + 1:
                changepoints.append(Changepoint(
                    index=t,
                    confidence=confidence,
                    type=cp_type
                ))
                last_alarm = t
                # Reset target to local mean after changepoint
                target = series[t]

        return changepoints

    @staticmethod
    def cumulative_sum(series: np.ndarray,
                       target: float | None = None) -> np.ndarray:
        """
        Compute the raw cumulative sum of deviations from target.

        S_t = Σ_{i=0}^{t} (x_i - μ)

        Useful for visualization. Changepoints appear as slope changes
        in the CUSUM chart.

        Args:
            series: 1D array
            target: Target value (mean of series if None)

        Returns:
            Cumulative sum array (same length as input)
        """
        series = np.asarray(series, dtype=float)
        if target is None:
            target = np.mean(series)
        return np.cumsum(series - target)


# ============================================================================
# PELT (Pruned Exact Linear Time)
# ============================================================================

class PELT:
    """
    Pruned Exact Linear Time algorithm for optimal changepoint detection.

    Finds the segmentation that minimizes total cost + penalty × number of
    changepoints. Uses dynamic programming with pruning to achieve O(n)
    average complexity (vs. O(n²) for exact).

    Default cost: negative log-likelihood under normal model (mean + variance
    per segment).
    """

    @staticmethod
    def detect(series: np.ndarray,
               penalty: str = 'bic',
               min_segment: int = 2,
               max_changepoints: int | None = None) -> ChangepointResult:
        """
        Detect changepoints via PELT.

        Args:
            series: 1D time series
            penalty: Penalty type — 'bic' (= ½ k ln n), 'aic' (= k),
                     or a float for custom penalty per changepoint
            min_segment: Minimum segment length
            max_changepoints: Optional cap on number of changepoints

        Returns:
            ChangepointResult with optimal segmentation
        """
        series = np.asarray(series, dtype=float)
        n = len(series)

        if n < 2 * min_segment:
            return ChangepointResult(
                changepoints=[],
                segments=[series.copy()],
                bic=PELT._segment_bic(series, 0)
            )

        # Compute penalty value
        if isinstance(penalty, str):
            k = 2  # parameters per segment (mean + variance)
            if penalty.lower() == 'bic':
                beta = 0.5 * k * np.log(n)
            elif penalty.lower() == 'aic':
                beta = float(k)
            else:
                raise ValueError(f"Unknown penalty type: {penalty}")
        else:
            beta = float(penalty)

        # Precompute cumulative statistics for O(1) segment cost
        cumsum = np.concatenate([[0], np.cumsum(series)])
        cumsum2 = np.concatenate([[0], np.cumsum(series ** 2)])

        def segment_cost(start: int, end: int) -> float:
            """Normal model cost for segment series[start:end]."""
            length = end - start
            if length < 1:
                return 0.0
            s1 = cumsum[end] - cumsum[start]
            s2 = cumsum2[end] - cumsum2[start]
            mean = s1 / length
            var = s2 / length - mean ** 2
            if var < 1e-15:
                var = 1e-15
            # -log-likelihood (up to constant)
            return length * np.log(var) + length

        # Dynamic programming with pruning
        # F[t] = optimal cost for series[0:t]
        F = np.full(n + 1, np.inf)
        F[0] = -beta  # offset so first segment doesn't double-count penalty

        # cp[t] = last changepoint before t in optimal partition
        cp = np.zeros(n + 1, dtype=int)

        # Candidate set for pruning
        candidates = {0}

        for t in range(min_segment, n + 1):
            best_cost = np.inf
            best_s = 0

            to_remove = set()

            for s in candidates:
                seg_len = t - s
                if seg_len < min_segment:
                    continue

                cost_s = F[s] + segment_cost(s, t) + beta
                if cost_s < best_cost:
                    best_cost = cost_s
                    best_s = s

                # Pruning: if F[s] + C(s,t) > F[t], s can never be optimal
                # for any future t' > t (PELT inequality)
                if F[s] + segment_cost(s, t) > best_cost + beta:
                    to_remove.add(s)

            F[t] = best_cost
            cp[t] = best_s
            candidates -= to_remove
            candidates.add(t)

        # Backtrack to find changepoints
        changepoint_indices = []
        t = n
        while t > 0:
            s = cp[t]
            if s > 0:
                changepoint_indices.append(s)
            t = s

        changepoint_indices.sort()

        # Apply max_changepoints limit
        if max_changepoints is not None and len(changepoint_indices) > max_changepoints:
            # Keep the ones with highest cost reduction
            # Approximate: keep evenly spaced
            step = len(changepoint_indices) / max_changepoints
            indices = [int(i * step) for i in range(max_changepoints)]
            changepoint_indices = [changepoint_indices[i] for i in indices]

        # Build segments and classify changepoints
        segments = []
        changepoints = []
        boundaries = [0] + changepoint_indices + [n]

        for i in range(len(boundaries) - 1):
            seg = series[boundaries[i]:boundaries[i + 1]]
            segments.append(seg)

        for idx in changepoint_indices:
            # Classify by comparing adjacent segments
            left_end = idx
            right_start = idx
            left_seg = series[max(0, left_end - min_segment * 2):left_end]
            right_seg = series[right_start:min(n, right_start + min_segment * 2)]

            if len(left_seg) > 0 and len(right_seg) > 0:
                cp_type = _classify_changepoint(left_seg, right_seg)
                # Confidence from the ratio of between-segment to within-segment variance
                conf = _changepoint_confidence(left_seg, right_seg)
            else:
                cp_type = 'mean_shift'
                conf = 0.5

            changepoints.append(Changepoint(index=idx, confidence=conf, type=cp_type))

        # Total BIC
        total_bic = PELT._segment_bic(series, len(changepoint_indices))

        return ChangepointResult(
            changepoints=changepoints,
            segments=segments,
            bic=total_bic
        )

    @staticmethod
    def _cost_function(segment: np.ndarray) -> float:
        """
        Normal model cost for a segment: n ln(σ²) where σ² = sample variance.

        This is the negative log-likelihood (up to additive constants) of
        observing the segment under a N(μ, σ²) model with MLE parameters.
        """
        n = len(segment)
        if n < 2:
            return 0.0
        var = np.var(segment, ddof=0)
        if var < 1e-15:
            var = 1e-15
        return n * np.log(var)

    @staticmethod
    def _segment_bic(series: np.ndarray, n_changepoints: int) -> float:
        """Compute BIC for the segmented model."""
        n = len(series)
        k = 2 * (n_changepoints + 1)  # mean + var per segment
        var = np.var(series, ddof=0)
        if var < 1e-15:
            var = 1e-15
        return n * np.log(var) + k * np.log(n)


# ============================================================================
# BAYESIAN ONLINE CHANGEPOINT DETECTION
# ============================================================================

class BayesianChangepoint:
    """
    Bayesian Online Changepoint Detection (Adams & MacKay, 2007).

    Maintains a posterior distribution over the run length r_t
    (time since last changepoint). At each step, the run length
    either resets to 0 (changepoint) or increments by 1.

    Uses a conjugate normal-inverse-gamma prior for tractable updates.
    """

    @staticmethod
    def detect(series: np.ndarray,
               prior_rate: float = 0.01,
               threshold: float = 0.5) -> list[Changepoint]:
        """
        Detect changepoints via Bayesian run length inference.

        Args:
            series: 1D time series
            prior_rate: Prior probability of a changepoint at each step (1/expected_duration)
            threshold: P(r_t = 0) threshold to declare a changepoint

        Returns:
            List of detected changepoints
        """
        series = np.asarray(series, dtype=float)
        n = len(series)

        run_length_dist = BayesianChangepoint.run_length_distribution(
            series, prior_rate
        )

        changepoints = []
        for t in range(1, n):
            # P(changepoint at t) = P(r_t = 0)
            p_cp = run_length_dist[t, 0]
            if p_cp > threshold:
                # Classify based on local behavior
                left = series[max(0, t - 20):t]
                right = series[t:min(n, t + 20)]
                if len(left) > 1 and len(right) > 1:
                    cp_type = _classify_changepoint(left, right)
                else:
                    cp_type = 'mean_shift'

                changepoints.append(Changepoint(
                    index=t,
                    confidence=min(1.0, float(p_cp)),
                    type=cp_type
                ))

        return changepoints

    @staticmethod
    def run_length_distribution(series: np.ndarray,
                                prior_rate: float = 0.01
                                ) -> np.ndarray:
        """
        Compute the full run length posterior distribution.

        P(r_t | x_{1:t}) for all t and all possible run lengths r.

        Uses Student-t predictive distribution from the conjugate
        normal-inverse-gamma model.

        Args:
            series: 1D time series
            prior_rate: Prior changepoint probability per step

        Returns:
            (n, n+1) array where [t, r] = P(run_length = r at time t)
        """
        series = np.asarray(series, dtype=float)
        n = len(series)

        # Run length probabilities: R[t, r] = P(r_t = r | x_{1:t})
        # For memory, we only keep what we need, but return full for API
        R = np.zeros((n, n + 1))
        R[0, 0] = 1.0

        # Sufficient statistics for normal-inverse-gamma conjugate
        # Prior hyperparameters
        mu0 = np.mean(series[:min(10, n)])
        kappa0 = 0.1
        alpha0 = 1.0
        beta0 = np.var(series[:min(10, n)], ddof=0) * alpha0
        if beta0 < 1e-10:
            beta0 = 1.0

        # Per-run-length sufficient stats
        # For efficiency, track arrays of sufficient stats
        max_run = n + 1
        kappa = np.full(max_run, kappa0)
        mu = np.full(max_run, mu0)
        alpha = np.full(max_run, alpha0)
        beta = np.full(max_run, beta0)

        for t in range(1, n):
            x = series[t]

            # Predictive probability under each run length
            # Student-t with 2*alpha degrees of freedom
            pred_probs = np.zeros(t + 1)
            for r in range(t):
                if R[t - 1, r] < 1e-300:
                    continue
                # Student-t parameters
                df = 2 * alpha[r]
                loc = mu[r]
                scale2 = beta[r] * (kappa[r] + 1) / (alpha[r] * kappa[r])
                scale = np.sqrt(max(scale2, 1e-15))

                pred_probs[r] = _student_t_pdf(x, df, loc, scale)

            # Growth probabilities: r_{t} = r_{t-1} + 1
            growth = R[t - 1, :t] * pred_probs[:t] * (1 - prior_rate)

            # Changepoint probability: r_t = 0
            cp_prob = np.sum(R[t - 1, :t] * pred_probs[:t] * prior_rate)

            # Normalize
            R[t, 0] = cp_prob
            R[t, 1:t + 1] = growth

            total = np.sum(R[t, :t + 1])
            if total > 1e-300:
                R[t, :t + 1] /= total

            # Update sufficient statistics for each run length
            # Shift: stats at position r+1 come from position r
            kappa_new = np.full(max_run, kappa0)
            mu_new = np.full(max_run, mu0)
            alpha_new = np.full(max_run, alpha0)
            beta_new = np.full(max_run, beta0)

            for r in range(min(t, max_run - 1)):
                # Update with new observation
                kappa_new[r + 1] = kappa[r] + 1
                mu_new[r + 1] = (kappa[r] * mu[r] + x) / kappa_new[r + 1]
                alpha_new[r + 1] = alpha[r] + 0.5
                beta_new[r + 1] = (beta[r] +
                                   0.5 * kappa[r] * (x - mu[r]) ** 2 / kappa_new[r + 1])

            kappa = kappa_new
            mu = mu_new
            alpha = alpha_new
            beta = beta_new

        return R


# ============================================================================
# SEGMENT STATISTICS
# ============================================================================

def segment_statistics(series: np.ndarray,
                       changepoints: list[Changepoint]) -> list[dict]:
    """
    Compute summary statistics for each segment defined by changepoints.

    Args:
        series: Full time series
        changepoints: List of changepoints (will be sorted by index)

    Returns:
        List of dicts with keys: start, end, mean, var, n, median, range
    """
    series = np.asarray(series, dtype=float)
    n = len(series)

    indices = sorted([cp.index for cp in changepoints])
    boundaries = [0] + indices + [n]

    stats = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        seg = series[start:end]

        if len(seg) == 0:
            continue

        stats.append({
            'start': start,
            'end': end,
            'n': len(seg),
            'mean': float(np.mean(seg)),
            'var': float(np.var(seg, ddof=1)) if len(seg) > 1 else 0.0,
            'std': float(np.std(seg, ddof=1)) if len(seg) > 1 else 0.0,
            'median': float(np.median(seg)),
            'min': float(np.min(seg)),
            'max': float(np.max(seg)),
            'range': float(np.max(seg) - np.min(seg)),
        })

    return stats


# ============================================================================
# INTERNAL UTILITIES
# ============================================================================

def _classify_changepoint(left: np.ndarray, right: np.ndarray) -> str:
    """
    Classify a changepoint as mean_shift, variance_shift, or trend_change.

    Heuristic based on comparing means, variances, and slopes.
    """
    if len(left) < 2 or len(right) < 2:
        return 'mean_shift'

    mean_left = np.mean(left)
    mean_right = np.mean(right)
    var_left = np.var(left, ddof=1)
    var_right = np.var(right, ddof=1)

    # Check for trend change via slope difference
    t_left = np.arange(len(left), dtype=float)
    t_right = np.arange(len(right), dtype=float)

    slope_left = _simple_slope(t_left, left)
    slope_right = _simple_slope(t_right, right)

    pooled_std = np.sqrt((var_left + var_right) / 2 + 1e-15)
    mean_diff = abs(mean_right - mean_left) / pooled_std
    var_ratio = max(var_left, var_right) / max(min(var_left, var_right), 1e-15)
    slope_diff = abs(slope_right - slope_left) / max(pooled_std, 1e-15)

    # Decision hierarchy
    if slope_diff > 1.0 and slope_diff > mean_diff:
        return 'trend_change'
    elif var_ratio > 2.0 and var_ratio > mean_diff:
        return 'variance_shift'
    else:
        return 'mean_shift'


def _changepoint_confidence(left: np.ndarray, right: np.ndarray) -> float:
    """
    Estimate confidence in a changepoint from adjacent segments.

    Uses the ratio of between-segment variance to within-segment variance
    (F-statistic-like).
    """
    if len(left) < 2 or len(right) < 2:
        return 0.5

    mean_left = np.mean(left)
    mean_right = np.mean(right)
    overall_mean = (np.sum(left) + np.sum(right)) / (len(left) + len(right))

    # Between-segment SS
    ss_between = (len(left) * (mean_left - overall_mean) ** 2 +
                  len(right) * (mean_right - overall_mean) ** 2)

    # Within-segment SS
    ss_within = np.sum((left - mean_left) ** 2) + np.sum((right - mean_right) ** 2)

    if ss_within < 1e-15:
        return 1.0 if ss_between > 1e-15 else 0.5

    f_ratio = ss_between / (ss_within / (len(left) + len(right) - 2))

    # Map F-ratio to [0, 1] confidence via sigmoid
    confidence = 1.0 / (1.0 + np.exp(-0.5 * (f_ratio - 4.0)))
    return float(np.clip(confidence, 0.0, 1.0))


def _simple_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Simple OLS slope."""
    n = len(x)
    if n < 2:
        return 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx < 1e-15:
        return 0.0
    return ss_xy / ss_xx


def _student_t_pdf(x: float, df: float, loc: float, scale: float) -> float:
    """
    Student's t PDF evaluated at x.

    p(x) = Γ((ν+1)/2) / (√(νπ) σ Γ(ν/2)) × (1 + ((x-μ)/σ)²/ν)^{-(ν+1)/2}

    Uses scipy when available, otherwise a numpy implementation.
    """
    if HAS_SCIPY:
        return float(scipy_norm.pdf(x, loc=loc, scale=scale))
        # Note: for proper Student-t, would use scipy.stats.t,
        # but normal approximation is adequate for large df

    # Numpy fallback: normal approximation (good for df > 30)
    z = (x - loc) / scale
    if df > 30:
        return float(np.exp(-0.5 * z ** 2) / (scale * np.sqrt(2 * np.pi)))

    # Proper Student-t via log-gamma
    half_df = 0.5 * df
    log_coeff = (_log_gamma(half_df + 0.5) - _log_gamma(half_df)
                 - 0.5 * np.log(df * np.pi) - np.log(scale))
    log_body = -(half_df + 0.5) * np.log(1.0 + z ** 2 / df)
    return float(np.exp(log_coeff + log_body))


def _log_gamma(x: float) -> float:
    """
    Log-gamma via Stirling's approximation.

    ln Γ(x) ≈ (x - 0.5) ln(x) - x + 0.5 ln(2π) + 1/(12x)

    Adequate for x > 1. For small x, use recursion: Γ(x) = Γ(x+1)/x.
    """
    if x < 1:
        # Recursion: Γ(x) = Γ(x+1) / x
        return _log_gamma(x + 1) - np.log(x)
    return ((x - 0.5) * np.log(x) - x + 0.5 * np.log(2 * np.pi) +
            1.0 / (12.0 * x))
