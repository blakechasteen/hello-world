"""
Ergodic Theory — Time Averages, Mixing, and Recurrence
=======================================================

Ergodic theory studies the long-term statistical behavior of
dynamical systems. A system is ergodic when time averages equal
space averages (Birkhoff's theorem).

Core Concepts:
- Birkhoff Ergodic Theorem: Time average = space average (a.e.)
- Mixing: Correlations decay → system "forgets" initial conditions
- Poincare Recurrence: Bounded systems return arbitrarily close to initial state
- Ergodicity Testing: Is the transition matrix ergodic?

Mathematical Foundation:
Birkhoff: lim_{N→∞} (1/N) Σ_{n=0}^{N-1} φ(Tⁿx) = ∫ φ dμ  a.e.
Mixing: lim_{n→∞} C(n) = lim_{n→∞} ⟨φ(Tⁿx)ψ(x)⟩ - ⟨φ⟩⟨ψ⟩ = 0
Poincare: E[τ_A] ≈ 1/μ(A) for recurrence time to set A

Applications to Warp Space:
- Verifying ergodicity of Thompson Sampling exploration
- Mixing rates in memory retrieval dynamics
- Recurrence analysis for scoring term cycles

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ErgodicResult:
    """Result of ergodic analysis.

    Attributes:
        time_average: Birkhoff time average
        variance: Variance of partial averages
        convergence_rate: Rate at which time average converges
        is_ergodic: Whether system appears ergodic
    """
    time_average: float
    variance: float = 0.0
    convergence_rate: float = 0.0
    is_ergodic: bool = True


@dataclass
class MixingResult:
    """Result of mixing analysis.

    Attributes:
        autocorrelation: Autocorrelation function C(τ) for each lag
        lags: Lag values
        decay_rate: Exponential decay rate (if mixing)
        mixing_time: Estimated mixing time
        is_mixing: Whether system appears mixing
    """
    autocorrelation: np.ndarray
    lags: np.ndarray
    decay_rate: float = 0.0
    mixing_time: float = np.inf
    is_mixing: bool = False


@dataclass
class RecurrenceResult:
    """Result of recurrence analysis.

    Attributes:
        mean_return_time: Average Poincare recurrence time
        return_times: Individual return times
        n_returns: Number of returns observed
        expected_return_time: Theoretical expected return time (1/μ(A))
    """
    mean_return_time: float
    return_times: np.ndarray
    n_returns: int = 0
    expected_return_time: float | None = None


# ============================================================================
# ERGODIC ANALYZER
# ============================================================================

class ErgodicAnalyzer:
    """Birkhoff ergodic theorem: time average of observables.

    For an ergodic system, the time average of any observable φ
    converges to the space average (ensemble average).

    Example:
        >>> ea = ErgodicAnalyzer()
        >>> # Trajectory from an ergodic system
        >>> trajectory = np.random.randn(10000, 2)
        >>> avg = ea.time_average(trajectory, lambda x: x[0]**2)
        >>> abs(avg - 1.0) < 0.1  # Should be ~1 for N(0,1)
        True
    """

    def time_average(
        self,
        trajectory: np.ndarray,
        observable: Callable[[np.ndarray], float],
    ) -> float:
        """Compute Birkhoff time average of an observable.

        Args:
            trajectory: State trajectory (n_steps, n_dims) or (n_steps,)
            observable: Function φ(x) -> scalar

        Returns:
            Time-averaged value of the observable
        """
        trajectory = np.asarray(trajectory, dtype=float)
        n = len(trajectory)

        total = 0.0
        for i in range(n):
            total += observable(trajectory[i])

        return total / n

    def analyze(
        self,
        trajectory: np.ndarray,
        observable: Callable[[np.ndarray], float],
        n_windows: int = 10,
    ) -> ErgodicResult:
        """Full ergodic analysis with convergence diagnostics.

        Splits trajectory into windows and checks whether partial
        averages converge.

        Args:
            trajectory: State trajectory (n_steps, n_dims)
            observable: Observable function
            n_windows: Number of windows for convergence check

        Returns:
            ErgodicResult with diagnostics
        """
        trajectory = np.asarray(trajectory, dtype=float)
        n = len(trajectory)

        # Compute observable values
        values = np.array([observable(trajectory[i]) for i in range(n)])

        # Overall time average
        time_avg = np.mean(values)

        # Partial averages for convergence check
        window_size = n // n_windows
        partial_avgs = []
        for w in range(n_windows):
            start = w * window_size
            end = start + window_size
            partial_avgs.append(np.mean(values[start:end]))

        partial_avgs = np.array(partial_avgs)
        variance = np.var(partial_avgs)

        # Convergence rate: how fast cumulative average stabilizes
        cum_avg = np.cumsum(values) / np.arange(1, n + 1)
        if n > 100:
            late_var = np.var(cum_avg[n // 2:])
            early_var = np.var(cum_avg[:n // 2])
            convergence_rate = early_var / (late_var + 1e-15)
        else:
            convergence_rate = 1.0

        # Ergodicity heuristic: partial averages should be consistent
        is_ergodic = variance < 0.1 * abs(time_avg) ** 2 + 1e-10

        return ErgodicResult(
            time_average=time_avg,
            variance=variance,
            convergence_rate=convergence_rate,
            is_ergodic=is_ergodic,
        )


# ============================================================================
# MIXING ANALYZER
# ============================================================================

class MixingAnalyzer:
    """Analyze mixing properties via autocorrelation decay.

    A mixing system has decaying autocorrelations: C(τ) → 0 as τ → ∞.
    The rate of decay characterizes how quickly the system "forgets"
    its initial state.

    Example:
        >>> ma = MixingAnalyzer()
        >>> # AR(1) process with φ=0.9
        >>> x = np.zeros(10000)
        >>> for i in range(1, len(x)):
        ...     x[i] = 0.9 * x[i-1] + np.random.randn()
        >>> result = ma.autocorrelation_decay(x, max_lag=100)
        >>> result.is_mixing
        True
    """

    def autocorrelation_decay(
        self,
        trajectory: np.ndarray,
        max_lag: int = 100,
        observable: Callable[[np.ndarray], float] | None = None,
    ) -> MixingResult:
        """Compute autocorrelation function and analyze decay.

        Args:
            trajectory: State trajectory (n_steps,) or (n_steps, n_dims)
            max_lag: Maximum lag τ to compute
            observable: Optional observable (default: first component or scalar)

        Returns:
            MixingResult with autocorrelation and decay analysis
        """
        trajectory = np.asarray(trajectory, dtype=float)
        n = len(trajectory)
        max_lag = min(max_lag, n // 4)

        # Extract scalar time series
        if observable is not None:
            series = np.array([observable(trajectory[i]) for i in range(n)])
        elif trajectory.ndim == 1:
            series = trajectory
        else:
            series = trajectory[:, 0]

        mean = np.mean(series)
        var = np.var(series)

        if var < 1e-15:
            return MixingResult(
                autocorrelation=np.zeros(max_lag + 1),
                lags=np.arange(max_lag + 1),
                is_mixing=False,
            )

        # Autocorrelation via direct computation
        lags = np.arange(max_lag + 1)
        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0

        centered = series - mean
        for lag in range(1, max_lag + 1):
            acf[lag] = np.mean(centered[:n - lag] * centered[lag:]) / var

        # Estimate exponential decay rate: C(τ) ~ exp(-τ/τ_mix)
        # Fit log|C(τ)| = -τ/τ_mix for positive ACF values
        positive_mask = acf[1:] > 0.01
        if np.sum(positive_mask) >= 3:
            pos_lags = lags[1:][positive_mask]
            log_acf = np.log(acf[1:][positive_mask])
            # Linear fit
            A = np.column_stack([pos_lags, np.ones(len(pos_lags))])
            try:
                coeffs = np.linalg.lstsq(A, log_acf, rcond=None)[0]
                decay_rate = -coeffs[0]
                mixing_time = 1.0 / (decay_rate + 1e-15) if decay_rate > 0 else np.inf
            except np.linalg.LinAlgError:
                decay_rate = 0.0
                mixing_time = np.inf
        else:
            decay_rate = 0.0
            mixing_time = np.inf

        is_mixing = decay_rate > 0.01 and mixing_time < n / 2

        return MixingResult(
            autocorrelation=acf,
            lags=lags,
            decay_rate=decay_rate,
            mixing_time=mixing_time,
            is_mixing=is_mixing,
        )


# ============================================================================
# RECURRENCE ANALYZER
# ============================================================================

class RecurrenceAnalyzer:
    """Poincare recurrence: expected return time to a neighborhood.

    By the Poincare recurrence theorem, a measure-preserving system
    returns to any set of positive measure. The expected return time
    is approximately 1/μ(A) where μ(A) is the measure of the set.

    Example:
        >>> ra = RecurrenceAnalyzer()
        >>> # 2D torus-like trajectory
        >>> trajectory = np.random.randn(10000, 2) * 0.5
        >>> result = ra.poincare_recurrence(trajectory, ball_radius=0.3)
        >>> result.mean_return_time > 0
        True
    """

    def poincare_recurrence(
        self,
        trajectory: np.ndarray,
        ball_radius: float,
        reference_point: np.ndarray | None = None,
    ) -> RecurrenceResult:
        """Measure Poincare recurrence time.

        Counts how many steps until the trajectory returns within
        ball_radius of a reference point.

        Args:
            trajectory: State trajectory (n_steps, n_dims) or (n_steps,)
            ball_radius: Radius of the recurrence ball
            reference_point: Center point (default: trajectory[0])

        Returns:
            RecurrenceResult with return time statistics
        """
        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.ndim == 1:
            trajectory = trajectory[:, None]

        n = len(trajectory)

        if reference_point is None:
            reference_point = trajectory[0]
        reference_point = np.asarray(reference_point, dtype=float)

        # Find all returns
        distances = np.linalg.norm(trajectory - reference_point, axis=1)
        in_ball = distances < ball_radius

        # Compute return times (gaps between visits)
        visit_indices = np.where(in_ball)[0]

        if len(visit_indices) < 2:
            return RecurrenceResult(
                mean_return_time=float(n),
                return_times=np.empty(0),
                n_returns=0,
            )

        # Skip initial consecutive visits (must leave before returning)
        return_times = []
        last_exit = visit_indices[0]
        was_inside = True

        for i in range(1, len(visit_indices)):
            gap = visit_indices[i] - visit_indices[i - 1]
            if gap > 1:  # Left the ball
                return_times.append(visit_indices[i] - last_exit)
                last_exit = visit_indices[i]

        if len(return_times) == 0:
            return RecurrenceResult(
                mean_return_time=float(n),
                return_times=np.empty(0),
                n_returns=0,
            )

        return_times = np.array(return_times, dtype=float)

        # Estimate expected return time from measure
        visit_fraction = np.sum(in_ball) / n
        expected = 1.0 / (visit_fraction + 1e-15) if visit_fraction > 0 else float(n)

        return RecurrenceResult(
            mean_return_time=float(np.mean(return_times)),
            return_times=return_times,
            n_returns=len(return_times),
            expected_return_time=expected,
        )


# ============================================================================
# ERGODICITY TEST FOR MARKOV CHAINS
# ============================================================================

def is_ergodic(transition_matrix: np.ndarray) -> bool:
    """Test if a Markov chain transition matrix is ergodic.

    A finite Markov chain is ergodic if it is irreducible and aperiodic.
    - Irreducible: every state reachable from every other state
    - Aperiodic: gcd of return times is 1 (sufficient: some Pᵢᵢ > 0)

    Args:
        transition_matrix: Row-stochastic matrix P (n, n)

    Returns:
        True if the chain is ergodic
    """
    P = np.asarray(transition_matrix, dtype=float)
    n = P.shape[0]

    if P.shape[0] != P.shape[1]:
        raise ValueError("Transition matrix must be square")

    # Check irreducibility via matrix power
    # P^n should have all positive entries for irreducible chain
    P_power = np.eye(n)
    for _ in range(n):
        P_power = P_power @ P

    if np.any(P_power < 1e-15):
        return False

    # Check aperiodicity: at least one state has positive self-transition
    # More robust: check that P^k has all positive entries for some k
    # If P^n already has all positive entries, it's aperiodic
    # (irreducible + some Pᵢᵢ > 0 is sufficient for aperiodicity)

    # Double-check with P^(n+1) — if both P^n and P^(n+1) are positive,
    # chain is definitely aperiodic
    P_power2 = P_power @ P
    if np.all(P_power2 > 1e-15):
        return True

    # Fallback: check eigenvalues — ergodic iff unique eigenvalue at 1
    eigenvalues = np.linalg.eigvals(P)
    unit_eigenvalues = np.sum(np.abs(np.abs(eigenvalues) - 1.0) < 1e-10)

    return unit_eigenvalues == 1
