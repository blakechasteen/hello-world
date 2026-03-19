"""
Survival Analysis — Time-to-Event Modeling
===========================================

Statistical methods for analyzing time until an event occurs,
handling censored observations (subjects who haven't experienced
the event yet).

Core Concepts:
- Kaplan-Meier: Nonparametric survival curve estimation
- Nelson-Aalen: Cumulative hazard estimation
- Cox Proportional Hazards: Semi-parametric regression
- Log-Rank Test: Comparing survival between groups

Mathematical Foundation:
Survival function: S(t) = P(T > t) = Π_{tᵢ≤t} (1 - dᵢ/nᵢ)  [KM]
Hazard function: h(t) = lim_{Δt→0} P(t ≤ T < t+Δt | T ≥ t) / Δt
Cumulative hazard: H(t) = Σ_{tᵢ≤t} dᵢ/nᵢ  [Nelson-Aalen]
Cox model: h(t|x) = h₀(t) exp(βᵀx)

Applications to Warp Space:
- Modeling time-to-convergence in weaving cycles
- Hazard rates for memory decay and retrieval failure
- Comparing survival of different policy strategies

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from dataclasses import dataclass

import numpy as np

try:
    from scipy.stats import chi2 as scipy_chi2
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SurvivalResult:
    """Result of survival analysis.

    Attributes:
        times: Unique event times (n_events,)
        survival_fn: Survival probability at each event time (n_events,)
        hazard_fn: Hazard rate at each event time (n_events,)
        cumulative_hazard: Cumulative hazard at each event time (n_events,)
        variance: Greenwood's variance estimate (n_events,)
        median_survival: Median survival time (may be None if not reached)
        at_risk: Number at risk at each time (n_events,)
    """
    times: np.ndarray
    survival_fn: np.ndarray
    hazard_fn: np.ndarray
    cumulative_hazard: np.ndarray
    variance: np.ndarray
    median_survival: float | None = None
    at_risk: np.ndarray | None = None


@dataclass
class CoxResult:
    """Result of Cox proportional hazards regression.

    Attributes:
        coefficients: Regression coefficients β (n_features,)
        standard_errors: Standard errors (n_features,)
        hazard_ratios: exp(β) (n_features,)
        log_partial_likelihood: Maximized partial log-likelihood
        concordance: Concordance index (C-statistic)
        n_iterations: Number of Newton-Raphson iterations
        converged: Whether optimization converged
    """
    coefficients: np.ndarray
    standard_errors: np.ndarray
    hazard_ratios: np.ndarray
    log_partial_likelihood: float = 0.0
    concordance: float = 0.0
    n_iterations: int = 0
    converged: bool = True


# ============================================================================
# KAPLAN-MEIER ESTIMATOR
# ============================================================================

class KaplanMeier:
    """Kaplan-Meier nonparametric survival estimator.

    The product-limit estimator of the survival function.
    Handles right-censored data.

    S(t) = Π_{tᵢ≤t} (1 - dᵢ/nᵢ)

    where dᵢ = deaths at time tᵢ, nᵢ = at risk at time tᵢ.

    Example:
        >>> km = KaplanMeier()
        >>> times = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        >>> events = np.array([1, 0, 1, 1, 0, 1, 1, 0])  # 1=event, 0=censored
        >>> result = km.fit(times, events)
        >>> result.median_survival
    """

    def fit(
        self,
        times: np.ndarray,
        events: np.ndarray,
    ) -> SurvivalResult:
        """Fit Kaplan-Meier estimator.

        Args:
            times: Observed times (n_subjects,)
            events: Event indicators, 1=event occurred, 0=censored (n_subjects,)

        Returns:
            SurvivalResult with survival curve
        """
        times = np.asarray(times, dtype=float)
        events = np.asarray(events, dtype=int)

        n = len(times)
        sort_idx = np.argsort(times)
        times_sorted = times[sort_idx]
        events_sorted = events[sort_idx]

        unique_event_times = np.unique(times_sorted[events_sorted == 1])

        survival = np.ones(len(unique_event_times))
        hazard = np.zeros(len(unique_event_times))
        variance = np.zeros(len(unique_event_times))
        at_risk_arr = np.zeros(len(unique_event_times), dtype=int)

        s_prev = 1.0
        var_sum = 0.0

        for i, t in enumerate(unique_event_times):
            # Number at risk: still alive and not censored before t
            n_at_risk = np.sum(times_sorted >= t)
            # Number of events at t
            n_events = np.sum((times_sorted == t) & (events_sorted == 1))

            at_risk_arr[i] = n_at_risk
            h_i = n_events / n_at_risk if n_at_risk > 0 else 0
            hazard[i] = h_i
            s_prev = s_prev * (1 - h_i)
            survival[i] = s_prev

            # Greenwood's formula
            if n_at_risk > 0 and n_at_risk != n_events:
                var_sum += n_events / (n_at_risk * (n_at_risk - n_events))
            variance[i] = s_prev ** 2 * var_sum

        # Cumulative hazard
        cumulative_hazard = -np.log(np.maximum(survival, 1e-15))

        # Median survival
        median = None
        below_half = np.where(survival <= 0.5)[0]
        if len(below_half) > 0:
            median = float(unique_event_times[below_half[0]])

        return SurvivalResult(
            times=unique_event_times,
            survival_fn=survival,
            hazard_fn=hazard,
            cumulative_hazard=cumulative_hazard,
            variance=variance,
            median_survival=median,
            at_risk=at_risk_arr,
        )


# ============================================================================
# NELSON-AALEN ESTIMATOR
# ============================================================================

class NelsonAalen:
    """Nelson-Aalen cumulative hazard estimator.

    H(t) = Σ_{tᵢ≤t} dᵢ/nᵢ

    Less biased than -log(KM) for small samples.

    Example:
        >>> na = NelsonAalen()
        >>> result = na.fit(times, events)
        >>> result.cumulative_hazard
    """

    def fit(
        self,
        times: np.ndarray,
        events: np.ndarray,
    ) -> SurvivalResult:
        """Fit Nelson-Aalen estimator.

        Args:
            times: Observed times (n_subjects,)
            events: Event indicators (n_subjects,)

        Returns:
            SurvivalResult with cumulative hazard emphasis
        """
        times = np.asarray(times, dtype=float)
        events = np.asarray(events, dtype=int)

        sort_idx = np.argsort(times)
        times_sorted = times[sort_idx]
        events_sorted = events[sort_idx]

        unique_event_times = np.unique(times_sorted[events_sorted == 1])

        cumhaz = np.zeros(len(unique_event_times))
        hazard = np.zeros(len(unique_event_times))
        variance = np.zeros(len(unique_event_times))
        at_risk_arr = np.zeros(len(unique_event_times), dtype=int)

        H = 0.0
        var_sum = 0.0

        for i, t in enumerate(unique_event_times):
            n_at_risk = np.sum(times_sorted >= t)
            n_events = np.sum((times_sorted == t) & (events_sorted == 1))

            at_risk_arr[i] = n_at_risk
            h_i = n_events / n_at_risk if n_at_risk > 0 else 0
            H += h_i
            hazard[i] = h_i
            cumhaz[i] = H

            if n_at_risk > 0:
                var_sum += n_events / (n_at_risk ** 2)
            variance[i] = var_sum

        # Survival from cumulative hazard: S(t) = exp(-H(t))
        survival = np.exp(-cumhaz)

        median = None
        below_half = np.where(survival <= 0.5)[0]
        if len(below_half) > 0:
            median = float(unique_event_times[below_half[0]])

        return SurvivalResult(
            times=unique_event_times,
            survival_fn=survival,
            hazard_fn=hazard,
            cumulative_hazard=cumhaz,
            variance=variance,
            median_survival=median,
            at_risk=at_risk_arr,
        )


# ============================================================================
# COX PROPORTIONAL HAZARDS
# ============================================================================

class CoxProportionalHazards:
    """Cox proportional hazards model (semi-parametric).

    h(t|x) = h₀(t) exp(βᵀx)

    Partial likelihood: L(β) = Π_{i: event} [exp(βᵀxᵢ) / Σ_{j∈R(tᵢ)} exp(βᵀxⱼ)]

    Example:
        >>> cox = CoxProportionalHazards()
        >>> X = np.random.randn(100, 3)
        >>> times = np.random.exponential(1, 100)
        >>> events = np.random.binomial(1, 0.7, 100)
        >>> result = cox.fit(X, times, events)
        >>> result.hazard_ratios
    """

    def fit(
        self,
        X: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> CoxResult:
        """Fit Cox proportional hazards model via Newton-Raphson.

        Args:
            X: Covariate matrix (n_subjects, n_features)
            times: Observed times (n_subjects,)
            events: Event indicators (n_subjects,)
            max_iter: Maximum Newton-Raphson iterations
            tol: Convergence tolerance on gradient norm

        Returns:
            CoxResult with coefficients and diagnostics
        """
        X = np.asarray(X, dtype=float)
        times = np.asarray(times, dtype=float)
        events = np.asarray(events, dtype=int)

        n, p = X.shape
        beta = np.zeros(p)

        sort_idx = np.argsort(-times)  # Sort descending
        X_s = X[sort_idx]
        times_s = times[sort_idx]
        events_s = events[sort_idx]

        converged = False
        ll = 0.0

        for iteration in range(max_iter):
            # Compute risk scores
            eta = X_s @ beta
            exp_eta = np.exp(eta - np.max(eta))  # Numerical stability

            # Cumulative sums (Breslow-style, ascending risk sets)
            # For descending-sorted data, cumsum gives the risk set sums
            risk_sum = np.cumsum(exp_eta)
            risk_sum_x = np.cumsum(exp_eta[:, None] * X_s, axis=0)
            risk_sum_xx = np.zeros((n, p, p))
            for i in range(n):
                xi = X_s[i:i + 1].T  # (p, 1)
                risk_sum_xx[i] = exp_eta[i] * (xi @ xi.T)
            risk_sum_xx = np.cumsum(risk_sum_xx, axis=0)

            # Gradient and Hessian
            gradient = np.zeros(p)
            hessian = np.zeros((p, p))
            ll = 0.0

            for i in range(n):
                if events_s[i] == 0:
                    continue

                rs = risk_sum[i]
                if rs < 1e-15:
                    continue

                rsx = risk_sum_x[i]
                rsxx = risk_sum_xx[i]

                gradient += X_s[i] - rsx / rs
                hessian -= rsxx / rs - np.outer(rsx, rsx) / (rs ** 2)
                ll += eta[i] - np.log(rs + 1e-15)

            grad_norm = np.linalg.norm(gradient)
            if grad_norm < tol:
                converged = True
                break

            try:
                delta = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                delta = gradient * 0.01
                logger.warning("Cox Hessian singular, using gradient step")

            beta -= delta

        # Standard errors from inverse Hessian
        try:
            inv_hessian = np.linalg.inv(-hessian)
            se = np.sqrt(np.maximum(np.diag(inv_hessian), 0))
        except np.linalg.LinAlgError:
            se = np.full(p, np.nan)

        # Concordance index
        concordance = self._concordance_index(X @ beta, times, events)

        return CoxResult(
            coefficients=beta,
            standard_errors=se,
            hazard_ratios=np.exp(beta),
            log_partial_likelihood=ll,
            concordance=concordance,
            n_iterations=iteration + 1,
            converged=converged,
        )

    @staticmethod
    def _concordance_index(
        risk_scores: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
    ) -> float:
        """Compute Harrell's concordance index (C-statistic)."""
        concordant = 0
        discordant = 0

        event_idx = np.where(events == 1)[0]
        for i in event_idx:
            # Compare with all subjects who survived longer
            longer = np.where(times > times[i])[0]
            for j in longer:
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] < risk_scores[j]:
                    discordant += 1

        total = concordant + discordant
        if total == 0:
            return 0.5
        return concordant / total


# ============================================================================
# LOG-RANK TEST
# ============================================================================

def log_rank_test(
    times1: np.ndarray,
    events1: np.ndarray,
    times2: np.ndarray,
    events2: np.ndarray,
) -> tuple[float, float]:
    """Log-rank test for comparing two survival curves.

    Tests H₀: S₁(t) = S₂(t) for all t.

    Args:
        times1, events1: Group 1 times and event indicators
        times2, events2: Group 2 times and event indicators

    Returns:
        (test_statistic, p_value) under chi-squared(1)
    """
    times1 = np.asarray(times1, dtype=float)
    events1 = np.asarray(events1, dtype=int)
    times2 = np.asarray(times2, dtype=float)
    events2 = np.asarray(events2, dtype=int)

    all_times = np.concatenate([times1, times2])
    all_events = np.concatenate([events1, events2])
    groups = np.concatenate([np.ones(len(times1)), np.zeros(len(times2))])

    unique_event_times = np.unique(all_times[all_events == 1])
    unique_event_times.sort()

    obs_minus_exp = 0.0
    variance = 0.0

    for t in unique_event_times:
        # At risk in each group
        n1 = np.sum(times1 >= t)
        n2 = np.sum(times2 >= t)
        n = n1 + n2

        # Events at t in each group
        d1 = np.sum((times1 == t) & (events1 == 1))
        d2 = np.sum((times2 == t) & (events2 == 1))
        d = d1 + d2

        if n < 2:
            continue

        # Expected events in group 1 under H₀
        e1 = d * n1 / n
        obs_minus_exp += d1 - e1

        # Variance under H₀
        if n > 1:
            variance += d * (n - d) * n1 * n2 / (n ** 2 * (n - 1))

    if variance < 1e-15:
        return 0.0, 1.0

    test_stat = obs_minus_exp ** 2 / variance

    # P-value from chi-squared(1)
    if HAS_SCIPY:
        p_value = 1 - scipy_chi2.cdf(test_stat, df=1)
    else:
        # Approximation: use normal distribution tail
        z = np.sqrt(test_stat)
        p_value = 2 * (1 - 0.5 * (1 + _erf_approx(z / np.sqrt(2))))

    return float(test_stat), float(p_value)


def _erf_approx(x: float) -> float:
    """Abramowitz & Stegun approximation to erf."""
    sign = np.sign(x)
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                 - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return sign * y
