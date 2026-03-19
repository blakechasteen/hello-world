"""
Chaos & Dynamical Systems — Strange Attractors, Predictability
==============================================================

Analysis of chaotic dynamical systems: detection, characterization,
and predictability estimation.

Core Concepts:
- Strange attractors: fractal geometric objects in phase space
- Correlation dimension: fractal dimension via Grassberger-Procaccia
- Takens embedding: reconstruct attractor from scalar time series
- Recurrence plots: binary matrix revealing dynamical structure
- Predictability horizon: time scale beyond which forecasts are useless

Mathematical Foundation:
Takens' theorem: if the attractor has dimension d, then a delay embedding
with dimension m >= 2d+1 is a diffeomorphism of the original attractor.

Correlation dimension: D_2 = lim_{r->0} log(C(r)) / log(r)
where C(r) = (2/N(N-1)) * #{(i,j) : ||x_i - x_j|| < r}

Predictability horizon: T_pred ~ (1/lambda_max) * ln(tolerance/delta_0)
where lambda_max is the maximum Lyapunov exponent.

Imports from stability_analysis.py: LyapunovExponents, LyapunovResult
Imports from bifurcation_analysis.py: BifurcationAnalyzer

Applications to Warp Space:
- Detecting chaotic regimes in Thompson Sampling dynamics
- Estimating forecast horizons for scoring term trajectories
- Phase space reconstruction from scalar performance metrics
- Recurrence analysis of weaving cycle patterns

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .stability_analysis import LyapunovExponents

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AttractorResult:
    """Result of strange attractor analysis.

    Attributes:
        dimension: Estimated correlation dimension (fractal dimension).
        embedding_dim: Embedding dimension used for reconstruction.
        lyapunov_spectrum: Array of Lyapunov exponents, sorted descending.
        predictability_horizon: Time steps before prediction error exceeds tolerance.
        is_chaotic: True if largest Lyapunov exponent is positive.
    """
    dimension: float
    embedding_dim: int
    lyapunov_spectrum: np.ndarray
    predictability_horizon: float
    is_chaotic: bool


# ============================================================================
# STANDARD TEST SYSTEMS
# ============================================================================

def lorenz_system(
    state: np.ndarray,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> np.ndarray:
    """Lorenz system: the canonical chaotic attractor.

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

    Args:
        state: [x, y, z] current state.
        sigma, rho, beta: System parameters.

    Returns:
        [dx/dt, dy/dt, dz/dt] derivatives.
    """
    x, y, z = state
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ])


def rossler_system(
    state: np.ndarray,
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
) -> np.ndarray:
    """Rossler system: simpler chaotic attractor.

    dx/dt = -y - z
    dy/dt = x + a*y
    dz/dt = b + z*(x - c)

    Args:
        state: [x, y, z] current state.
        a, b, c: System parameters.

    Returns:
        [dx/dt, dy/dt, dz/dt] derivatives.
    """
    x, y, z = state
    return np.array([
        -y - z,
        x + a * y,
        b + z * (x - c),
    ])


def henon_map(state: np.ndarray, a: float = 1.4, b: float = 0.3) -> np.ndarray:
    """Henon map: canonical 2D chaotic map.

    x_{n+1} = 1 - a*x_n^2 + y_n
    y_{n+1} = b*x_n

    Args:
        state: [x, y] current state.
        a, b: Map parameters.

    Returns:
        [x_next, y_next].
    """
    x, y = state
    return np.array([
        1.0 - a * x**2 + y,
        b * x,
    ])


def _integrate_ode(
    f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    dt: float,
    n_steps: int,
    transient: int = 0,
) -> np.ndarray:
    """Simple RK4 integrator for generating trajectories.

    Args:
        f: System dynamics dx/dt = f(x).
        x0: Initial condition.
        dt: Time step.
        n_steps: Total integration steps.
        transient: Number of initial steps to discard.

    Returns:
        Trajectory of shape (n_steps - transient, dim).
    """
    dim = len(x0)
    total = n_steps + transient
    trajectory = np.zeros((total, dim))
    trajectory[0] = x0

    for i in range(1, total):
        x = trajectory[i - 1]
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        trajectory[i] = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return trajectory[transient:]


def _iterate_map(
    f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    n_steps: int,
    transient: int = 0,
) -> np.ndarray:
    """Iterate a discrete map to generate trajectory.

    Args:
        f: Map x_{n+1} = f(x_n).
        x0: Initial condition.
        n_steps: Total iterations.
        transient: Initial iterations to discard.

    Returns:
        Trajectory of shape (n_steps - transient, dim).
    """
    dim = len(x0)
    total = n_steps + transient
    trajectory = np.zeros((total, dim))
    trajectory[0] = x0

    for i in range(1, total):
        trajectory[i] = f(trajectory[i - 1])

    return trajectory[transient:]


# ============================================================================
# STRANGE ATTRACTOR DETECTOR
# ============================================================================

class StrangeAttractorDetector:
    """Detect and characterize strange attractors from time series data.

    Provides tools for phase space reconstruction, dimension estimation,
    and recurrence analysis of dynamical systems.
    """

    @staticmethod
    def correlation_dimension(
        trajectory: np.ndarray,
        max_dim: int = 10,
        r_range: tuple[float, float] | None = None,
        n_ref: int = 500,
    ) -> float:
        """Estimate correlation dimension via Grassberger-Procaccia algorithm.

        D_2 = lim_{r->0} d(log C(r)) / d(log r)

        where C(r) = (2 / N(N-1)) * #{pairs with distance < r}

        Args:
            trajectory: Trajectory in phase space, shape (N, d).
            max_dim: Maximum dimension for saturation check (unused in
                basic version, kept for API consistency).
            r_range: (r_min, r_max) range for regression. If None,
                auto-selected from data.
            n_ref: Number of reference points to subsample (for speed).

        Returns:
            Estimated correlation dimension D_2.
        """
        trajectory = np.asarray(trajectory)
        N = trajectory.shape[0]

        # Subsample reference points if needed
        if N > n_ref:
            rng = np.random.default_rng(42)
            idx = rng.choice(N, n_ref, replace=False)
            ref_points = trajectory[idx]
        else:
            ref_points = trajectory

        # Compute all pairwise distances from reference points to all points
        # Use broadcasting: (n_ref, 1, d) - (1, N, d)
        diffs = ref_points[:, None, :] - trajectory[None, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))  # (n_ref, N)

        # Flatten and remove self-distances (zero)
        all_dists = dists.flatten()
        all_dists = all_dists[all_dists > 0]

        if len(all_dists) == 0:
            return 0.0

        # Auto-select r_range
        if r_range is None:
            r_min = np.percentile(all_dists, 1)
            r_max = np.percentile(all_dists, 50)
            r_min = max(r_min, 1e-10)
        else:
            r_min, r_max = r_range

        # Compute C(r) at multiple scales
        n_scales = 30
        r_vals = np.logspace(np.log10(r_min), np.log10(r_max), n_scales)
        n_pairs = len(all_dists)
        c_vals = np.array([np.sum(all_dists < r) / n_pairs for r in r_vals])

        # Remove zeros
        mask = c_vals > 0
        if mask.sum() < 3:
            return 0.0

        log_r = np.log(r_vals[mask])
        log_c = np.log(c_vals[mask])

        # Linear regression for slope = D_2
        # Use middle portion for best scaling region
        n_valid = len(log_r)
        start = n_valid // 5
        end = 4 * n_valid // 5
        if end - start < 3:
            start = 0
            end = n_valid

        log_r_fit = log_r[start:end]
        log_c_fit = log_c[start:end]

        # Least squares: slope = D_2
        A = np.vstack([log_r_fit, np.ones(len(log_r_fit))]).T
        result = np.linalg.lstsq(A, log_c_fit, rcond=None)
        D_2 = result[0][0]

        return float(max(0.0, D_2))

    @staticmethod
    def recurrence_plot(
        trajectory: np.ndarray,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Compute recurrence plot: binary matrix of phase space recurrences.

        R(i,j) = 1 if ||x_i - x_j|| < threshold, else 0.

        Recurrence plots reveal:
        - Periodic orbits: diagonal lines parallel to main diagonal
        - Chaos: short diagonal lines with gaps
        - Drift: fading from diagonal
        - Intermittency: white bands

        Args:
            trajectory: Phase space trajectory, shape (N, d).
            threshold: Distance threshold epsilon. If None, uses
                10% of the mean pairwise distance.

        Returns:
            Binary recurrence matrix of shape (N, N).
        """
        trajectory = np.asarray(trajectory)
        N = trajectory.shape[0]

        # Pairwise distances
        diffs = trajectory[:, None, :] - trajectory[None, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))

        if threshold is None:
            threshold = 0.1 * np.mean(dists)

        return (dists < threshold).astype(np.int8)

    @staticmethod
    def takens_embedding(
        series: np.ndarray,
        delay: int | None = None,
        dimension: int | None = None,
    ) -> np.ndarray:
        """Delay coordinate embedding (Takens' theorem).

        Reconstructs phase space from scalar time series:
        x_i -> [x_i, x_{i+tau}, x_{i+2*tau}, ..., x_{i+(m-1)*tau}]

        Takens' theorem guarantees this is a diffeomorphism of the
        original attractor if m >= 2*d_A + 1 where d_A is the
        attractor dimension.

        Args:
            series: 1D time series, shape (N,).
            delay: Time delay tau. If None, uses mutual_information_delay.
            dimension: Embedding dimension m. If None, uses
                false_nearest_neighbors.

        Returns:
            Embedded trajectory, shape (N - (m-1)*tau, m).
        """
        series = np.asarray(series).flatten()
        N = len(series)

        if delay is None:
            delay = StrangeAttractorDetector.mutual_information_delay(series)
        if dimension is None:
            dimension = StrangeAttractorDetector.false_nearest_neighbors(
                series, delay
            )

        # Construct delay vectors
        n_vectors = N - (dimension - 1) * delay
        if n_vectors <= 0:
            raise ValueError(
                f"Series too short ({N}) for delay={delay}, dim={dimension}. "
                f"Need at least {(dimension - 1) * delay + 1} points."
            )

        embedded = np.zeros((n_vectors, dimension))
        for d in range(dimension):
            embedded[:, d] = series[d * delay: d * delay + n_vectors]

        return embedded

    @staticmethod
    def mutual_information_delay(
        series: np.ndarray,
        max_delay: int = 50,
        n_bins: int = 16,
    ) -> int:
        """Find optimal delay for Takens embedding via MI minimum.

        The first local minimum of mutual information I(x_t, x_{t+tau})
        gives a good delay: components are maximally independent while
        still sharing attractor structure.

        Args:
            series: 1D time series, shape (N,).
            max_delay: Maximum delay to test.
            n_bins: Number of histogram bins.

        Returns:
            Optimal delay tau (first local minimum of MI).
        """
        series = np.asarray(series).flatten()
        N = len(series)
        max_delay = min(max_delay, N // 4)

        mi_values = np.zeros(max_delay)

        for tau in range(max_delay):
            x = series[:N - tau] if tau > 0 else series
            y = series[tau:] if tau > 0 else series

            # Joint histogram
            hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
            p_joint = hist_2d / hist_2d.sum()

            # Marginals
            p_x = p_joint.sum(axis=1)
            p_y = p_joint.sum(axis=0)

            # MI = sum p(x,y) log(p(x,y) / (p(x)*p(y)))
            mi = 0.0
            for i in range(n_bins):
                for j in range(n_bins):
                    if p_joint[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                        mi += p_joint[i, j] * np.log(
                            p_joint[i, j] / (p_x[i] * p_y[j])
                        )
            mi_values[tau] = mi

        # Find first local minimum (skip tau=0)
        for tau in range(1, max_delay - 1):
            if mi_values[tau] < mi_values[tau - 1] and mi_values[tau] < mi_values[tau + 1]:
                return tau

        # Fallback: use tau where MI drops to 1/e of initial
        threshold = mi_values[0] / np.e
        below = np.where(mi_values < threshold)[0]
        if len(below) > 0:
            return int(below[0])

        return max(1, max_delay // 4)

    @staticmethod
    def false_nearest_neighbors(
        series: np.ndarray,
        delay: int,
        max_dim: int = 10,
        threshold: float = 15.0,
        n_ref: int = 500,
    ) -> int:
        """Determine optimal embedding dimension via false nearest neighbors.

        A neighbor is "false" if it's close in dimension d but far in
        dimension d+1. When the FNN fraction drops below a threshold,
        the attractor is unfolded.

        Args:
            series: 1D time series, shape (N,).
            delay: Time delay tau for embedding.
            max_dim: Maximum dimension to test.
            threshold: Distance ratio threshold for false neighbor detection.
            n_ref: Number of reference points to test.

        Returns:
            Optimal embedding dimension m.
        """
        series = np.asarray(series).flatten()
        N = len(series)

        for dim in range(1, max_dim + 1):
            n_vectors = N - dim * delay
            if n_vectors < n_ref:
                return dim

            # Build embedding at dimension dim
            embedded = np.zeros((n_vectors, dim))
            for d in range(dim):
                embedded[:, d] = series[d * delay: d * delay + n_vectors]

            # Extra coordinate for dim+1
            n_vectors_next = N - (dim) * delay
            if n_vectors_next < 2:
                return dim

            extra = series[dim * delay: dim * delay + n_vectors_next]

            # Subsample reference points
            rng = np.random.default_rng(42)
            n_test = min(n_ref, n_vectors_next)
            test_idx = rng.choice(n_vectors_next, n_test, replace=False)

            n_false = 0
            n_total = 0

            for idx in test_idx:
                # Find nearest neighbor in dim-dimensional embedding
                point = embedded[idx]
                dists = np.sqrt(np.sum((embedded[:n_vectors_next] - point)**2, axis=1))
                dists[idx] = np.inf  # Exclude self

                nn_idx = np.argmin(dists)
                nn_dist = dists[nn_idx]

                if nn_dist < 1e-15:
                    continue

                n_total += 1

                # Check if neighbor is false: distance in extra dimension
                extra_dist = abs(extra[idx] - extra[nn_idx])
                ratio = extra_dist / nn_dist

                if ratio > threshold:
                    n_false += 1

            if n_total > 0:
                fnn_fraction = n_false / n_total
                if fnn_fraction < 0.01:
                    return dim

        return max_dim


# ============================================================================
# PREDICTABILITY ANALYZER
# ============================================================================

class PredictabilityAnalyzer:
    """Estimate prediction horizons and forecast skill of chaotic systems.

    Uses Lyapunov exponents to bound theoretical predictability, and
    empirical forecast skill to measure practical predictability.
    """

    @staticmethod
    def horizon(
        lyapunov_max: float,
        tolerance: float = 0.01,
        initial_error: float = 1e-8,
    ) -> float:
        """Theoretical prediction horizon from Lyapunov exponent.

        T_pred = (1 / lambda_max) * ln(tolerance / initial_error)

        For a chaotic system, errors grow as delta(t) ~ delta(0) * exp(lambda_max * t).
        The prediction horizon is when errors reach the tolerance level.

        Args:
            lyapunov_max: Largest Lyapunov exponent (must be positive for chaos).
            tolerance: Acceptable error level (fraction of attractor diameter).
            initial_error: Initial measurement/estimation error.

        Returns:
            Prediction horizon in time units. Returns inf if lambda_max <= 0
            (non-chaotic system has infinite predictability).
        """
        if lyapunov_max <= 0:
            return float('inf')
        if tolerance <= initial_error:
            return 0.0

        return (1.0 / lyapunov_max) * np.log(tolerance / initial_error)

    @staticmethod
    def forecast_skill(
        trajectory: np.ndarray,
        model_fn: Callable[[np.ndarray], np.ndarray],
        horizons: np.ndarray,
        n_trials: int = 50,
    ) -> np.ndarray:
        """Measure empirical forecast skill at various prediction horizons.

        For each horizon h, computes the correlation between actual
        trajectory and model predictions started from the same initial
        conditions. Skill = 1.0 means perfect, 0.0 means no skill.

        Args:
            trajectory: True trajectory, shape (N, d).
            model_fn: Single-step predictor x_{t+1} = model_fn(x_t).
            horizons: Array of prediction horizons to test.
            n_trials: Number of starting points to average over.

        Returns:
            Array of forecast skill values, shape (len(horizons),), in [0, 1].
        """
        trajectory = np.asarray(trajectory)
        N, d = trajectory.shape
        horizons = np.asarray(horizons, dtype=int)
        max_horizon = int(horizons.max())

        rng = np.random.default_rng(42)

        # Starting indices (must leave room for max horizon)
        max_start = N - max_horizon - 1
        if max_start < 1:
            return np.zeros(len(horizons))

        n_trials = min(n_trials, max_start)
        starts = rng.choice(max_start, n_trials, replace=False)

        skills = np.zeros(len(horizons))

        for h_idx, h in enumerate(horizons):
            if h <= 0:
                skills[h_idx] = 1.0
                continue

            correlations = []
            for start in starts:
                # True future state
                true_state = trajectory[start + h]

                # Predicted future state by iterating model
                pred_state = trajectory[start].copy()
                valid = True
                for _ in range(h):
                    try:
                        pred_state = model_fn(pred_state)
                        if not np.all(np.isfinite(pred_state)):
                            valid = False
                            break
                    except Exception:
                        valid = False
                        break

                if valid:
                    # Normalized correlation (cosine similarity as skill)
                    norm_true = np.linalg.norm(true_state)
                    norm_pred = np.linalg.norm(pred_state)
                    if norm_true > 1e-15 and norm_pred > 1e-15:
                        corr = np.dot(true_state, pred_state) / (norm_true * norm_pred)
                        correlations.append(max(0.0, corr))

            if correlations:
                skills[h_idx] = float(np.mean(correlations))

        return skills


# ============================================================================
# CONVENIENCE: FULL ATTRACTOR ANALYSIS
# ============================================================================

def analyze_attractor(
    trajectory: np.ndarray,
    dt: float = 1.0,
) -> AttractorResult:
    """Full attractor analysis pipeline.

    Computes dimension, Lyapunov spectrum, and predictability from
    a phase space trajectory.

    Args:
        trajectory: Phase space trajectory, shape (N, d).
        dt: Time step between points.

    Returns:
        AttractorResult with dimension, exponents, horizon, chaos flag.
    """
    trajectory = np.asarray(trajectory)
    d = trajectory.shape[1]

    # Correlation dimension
    dim = StrangeAttractorDetector.correlation_dimension(trajectory)

    # Lyapunov exponents from trajectory
    lyap_result = LyapunovExponents.compute_from_trajectory(trajectory, dt=dt)

    # Predictability horizon
    lambda_max = lyap_result.max_exponent
    horizon = PredictabilityAnalyzer.horizon(lambda_max)

    return AttractorResult(
        dimension=dim,
        embedding_dim=d,
        lyapunov_spectrum=lyap_result.exponents,
        predictability_horizon=horizon,
        is_chaotic=lyap_result.is_chaotic,
    )


if __name__ == "__main__":
    print("Chaos & Dynamical Systems Module Demo")
    print("=" * 50)

    # Generate Lorenz trajectory
    x0 = np.array([1.0, 1.0, 1.0])
    traj = _integrate_ode(lorenz_system, x0, dt=0.01, n_steps=5000, transient=1000)
    print(f"Lorenz trajectory: {traj.shape}")

    # Correlation dimension (Lorenz ~ 2.06)
    D2 = StrangeAttractorDetector.correlation_dimension(traj)
    print(f"Correlation dimension: {D2:.2f} (expected ~2.06)")

    # Delay embedding from x-component
    x_series = traj[:, 0]
    tau = StrangeAttractorDetector.mutual_information_delay(x_series)
    print(f"Optimal delay: {tau}")

    m = StrangeAttractorDetector.false_nearest_neighbors(x_series, tau)
    print(f"Optimal embedding dimension: {m}")

    embedded = StrangeAttractorDetector.takens_embedding(x_series, delay=tau, dimension=m)
    print(f"Embedded shape: {embedded.shape}")

    # Recurrence plot (small subset)
    rp = StrangeAttractorDetector.recurrence_plot(traj[:200])
    print(f"Recurrence plot: {rp.shape}, density: {rp.mean():.3f}")

    # Henon map
    henon_traj = _iterate_map(henon_map, np.array([0.1, 0.1]), n_steps=2000, transient=500)
    D2_henon = StrangeAttractorDetector.correlation_dimension(henon_traj)
    print(f"\nHenon dimension: {D2_henon:.2f} (expected ~1.22)")

    # Predictability
    print(f"\nPredictability horizon (lambda=0.9): {PredictabilityAnalyzer.horizon(0.9):.1f} steps")
    print(f"Predictability horizon (lambda=0.0): {PredictabilityAnalyzer.horizon(0.0)} steps")

    print("\n" + "=" * 50)
    print("Chaos & dynamical systems module ready!")
