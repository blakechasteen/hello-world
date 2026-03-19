r"""
Topological Entropy — KAM Theory, Topological Entropy, Lyapunov Spectrum
=========================================================================

Measures of dynamical complexity: topological entropy quantifies orbit
growth rate, KAM theory determines persistence of invariant tori under
perturbation, and the full Lyapunov spectrum reveals all dimensions
of stretching and compression.

Core Concepts:
- KAM Theory: Quasi-periodic orbits persist under small perturbations if
  frequencies satisfy Diophantine conditions
- Topological Entropy: h_top = lim_{ε→0} lim_{T→∞} (1/T) log N(ε,T)
- Lyapunov Spectrum: All exponents {λ₁ ≥ λ₂ ≥ ... ≥ λ_d} from QR decomposition

Mathematical Foundation:
KAM condition: |ω·k| ≥ γ/|k|^τ for all k ∈ Z^d \ {0}
ε-entropy: H(ε,T) = log(min covers of T-length orbit segments by ε-balls)
Lyapunov: λ_i = lim_{t→∞} (1/t) log σ_i(Φ(t)) where Φ is fundamental matrix

Applications to Warp Space:
- Detecting chaos vs quasi-periodicity in iterative weaving cycles
- Measuring information production rate of dynamical memory processes
- Full spectral characterization of embedding flow stability
- KAM persistence analysis for perturbed scoring term dynamics

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
class KAMResult:
    """Result of KAM theory analysis.

    Attributes:
        is_persistent: Whether invariant torus persists under perturbation.
        diophantine_constant: γ in |ω·k| ≥ γ/|k|^τ.
        diophantine_exponent: τ in the Diophantine condition.
        small_divisors: Smallest |ω·k| values encountered.
        perturbation_bound: Maximum perturbation strength for persistence.
    """
    is_persistent: bool
    diophantine_constant: float
    diophantine_exponent: float
    small_divisors: np.ndarray
    perturbation_bound: float


@dataclass
class TopologicalEntropyResult:
    """Result of topological entropy computation.

    Attributes:
        entropy: Estimated topological entropy h_top.
        epsilon_values: ε values used in computation.
        count_values: Number of ε-distinguishable orbit segments per ε.
        correlation_dimension: Estimated from scaling of counts.
    """
    entropy: float
    epsilon_values: np.ndarray
    count_values: np.ndarray
    correlation_dimension: float


@dataclass
class LyapunovSpectrumResult:
    """Full Lyapunov spectrum result.

    Attributes:
        exponents: All Lyapunov exponents sorted descending (d,).
        kaplan_yorke_dimension: D_KY from Kaplan-Yorke conjecture.
        kolmogorov_sinai_entropy: h_KS = Σ max(λ_i, 0) (Pesin formula).
        sum_exponents: Σ λ_i (related to phase space contraction).
        convergence_history: History of exponent estimates (n_steps, d).
    """
    exponents: np.ndarray
    kaplan_yorke_dimension: float
    kolmogorov_sinai_entropy: float
    sum_exponents: float
    convergence_history: np.ndarray


# ============================================================================
# KAM THEORY
# ============================================================================

class KAMTheory:
    """Kolmogorov-Arnold-Moser (KAM) theory analysis.

    Determines whether quasi-periodic orbits (invariant tori) persist
    under small perturbations of an integrable Hamiltonian system.

    The key condition is Diophantine: the frequency vector ω must be
    sufficiently irrational, satisfying |ω·k| ≥ γ/|k|^τ for all
    integer vectors k ≠ 0.

    Example:
        >>> kam = KAMTheory()
        >>> H0 = lambda J: 0.5 * np.sum(J**2)
        >>> omega = np.array([1.0, (1 + np.sqrt(5))/2])  # Golden ratio
        >>> result = kam.check_invariant_torus(H0, omega, perturbation=0.01)
        >>> result.is_persistent  # True (golden ratio is well-irrational)
    """

    def check_invariant_torus(
        self,
        hamiltonian: Callable[[np.ndarray], float],
        frequency: np.ndarray,
        perturbation: float = 0.01,
        max_order: int = 20,
        tau: float | None = None,
    ) -> KAMResult:
        """Check persistence of invariant torus with given frequency.

        Tests the Diophantine condition and estimates whether the
        perturbation is within the KAM stability bound.

        Args:
            hamiltonian: Integrable Hamiltonian H₀(J).
            frequency: Frequency vector ω = ∂H₀/∂J.
            perturbation: Perturbation strength ε.
            max_order: Maximum |k| to check in Diophantine condition.
            tau: Diophantine exponent (default: d-1+δ for small δ).

        Returns:
            KAMResult with persistence analysis.
        """
        omega = np.asarray(frequency, dtype=float)
        d = len(omega)

        if tau is None:
            tau = float(d)  # Standard choice: d (slightly above d-1)

        # Check Diophantine condition
        gamma, small_divisors = self._check_diophantine(omega, max_order, tau)

        # KAM perturbation bound (simplified estimate)
        # The actual bound depends on the Hamiltonian structure
        # Rough estimate: ε_KAM ∝ γ^2 / (norm of ω)^{2τ}
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 0:
            eps_bound = gamma ** 2 / (omega_norm ** (2 * tau) + 1e-300)
        else:
            eps_bound = 0.0

        is_persistent = (perturbation < eps_bound) and (gamma > 0)

        return KAMResult(
            is_persistent=is_persistent,
            diophantine_constant=gamma,
            diophantine_exponent=tau,
            small_divisors=small_divisors,
            perturbation_bound=eps_bound,
        )

    def _check_diophantine(
        self, omega: np.ndarray, max_order: int, tau: float
    ) -> tuple[float, np.ndarray]:
        """Check Diophantine condition and compute the constant γ.

        Finds γ = min_{k ≠ 0} |ω·k| · |k|^τ over integer vectors
        k with |k| ≤ max_order.

        Returns:
            (gamma, small_divisors) where small_divisors are the
            smallest |ω·k| values.
        """
        d = len(omega)
        small_divisors = []
        min_gamma = np.inf

        # Generate integer vectors k with 1 ≤ |k|_1 ≤ max_order
        for order in range(1, max_order + 1):
            for k in self._integer_vectors(d, order):
                dot = abs(np.dot(omega, k))
                k_norm = np.sum(np.abs(k))
                gamma_k = dot * k_norm ** tau

                min_gamma = min(min_gamma, gamma_k)
                if dot < 0.1:
                    small_divisors.append(dot)

        if min_gamma == np.inf:
            min_gamma = 0.0

        small_divisors = np.array(sorted(small_divisors)[:10]) if small_divisors else np.array([])
        return float(min_gamma), small_divisors

    @staticmethod
    def _integer_vectors(d: int, max_norm: int) -> list[np.ndarray]:
        """Generate integer vectors with L1 norm exactly max_norm.

        For efficiency, samples rather than enumerating all vectors
        when dimension is high.
        """
        if d == 1:
            return [np.array([max_norm]), np.array([-max_norm])]

        vectors = []
        if d == 2:
            for k1 in range(-max_norm, max_norm + 1):
                k2 = max_norm - abs(k1)
                if k2 >= 0:
                    if k2 > 0:
                        vectors.append(np.array([k1, k2]))
                        vectors.append(np.array([k1, -k2]))
                    else:
                        vectors.append(np.array([k1, 0]))
        else:
            # Sample for higher dimensions
            n_samples = min(100, 3 ** d)
            for _ in range(n_samples):
                k = np.random.randint(-max_norm, max_norm + 1, size=d)
                if np.sum(np.abs(k)) > 0:
                    vectors.append(k)

        return vectors

    def frequency_analysis(self, trajectory: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Extract dominant frequencies from a trajectory via FFT.

        Useful for determining the frequency vector ω of a quasi-periodic
        orbit from numerical integration data.

        Args:
            trajectory: Time series (T, d).
            dt: Time step.

        Returns:
            Dominant frequency for each dimension (d,).
        """
        trajectory = np.atleast_2d(trajectory)
        if trajectory.shape[0] < trajectory.shape[1]:
            trajectory = trajectory.T
        T, d = trajectory.shape

        frequencies = np.zeros(d)
        for j in range(d):
            fft = np.fft.rfft(trajectory[:, j] - np.mean(trajectory[:, j]))
            power = np.abs(fft) ** 2
            freqs = np.fft.rfftfreq(T, d=dt)
            # Dominant frequency (excluding DC)
            if len(power) > 1:
                idx = np.argmax(power[1:]) + 1
                frequencies[j] = freqs[idx]

        return frequencies


# ============================================================================
# TOPOLOGICAL ENTROPY
# ============================================================================

class TopologicalEntropy:
    """Topological entropy estimation from trajectory data.

    h_top measures the exponential growth rate of distinguishable
    orbit segments. For a map f, it equals the growth rate of the
    number of ε-separated n-orbits as n → ∞.

    Higher h_top means more complex dynamics and greater unpredictability.
    h_top = 0 for periodic/quasi-periodic orbits, h_top > 0 for chaos.

    Example:
        >>> te = TopologicalEntropy()
        >>> # Logistic map at r=4 (fully chaotic, h_top = log(2))
        >>> x = np.zeros(10000)
        >>> x[0] = 0.1
        >>> for i in range(9999): x[i+1] = 4*x[i]*(1-x[i])
        >>> result = te.compute(x.reshape(-1,1), epsilon_range=np.logspace(-3, -1, 20))
    """

    def compute(
        self,
        trajectory: np.ndarray,
        epsilon_range: np.ndarray | None = None,
        max_orbit_length: int = 50,
    ) -> TopologicalEntropyResult:
        """Estimate topological entropy from trajectory.

        Uses ε-entropy approach: count ε-distinguishable orbit segments
        of length T, fit growth rate as T → ∞.

        Args:
            trajectory: Trajectory data (T, d).
            epsilon_range: Range of ε values for multi-scale analysis.
            max_orbit_length: Maximum orbit segment length.

        Returns:
            TopologicalEntropyResult.
        """
        trajectory = np.atleast_2d(trajectory)
        if trajectory.shape[0] < trajectory.shape[1]:
            trajectory = trajectory.T
        T, d = trajectory.shape

        if epsilon_range is None:
            # Auto-determine from data scale
            std = np.std(trajectory, axis=0).mean()
            if std == 0:
                std = 1.0
            epsilon_range = np.logspace(np.log10(0.01 * std), np.log10(std), 15)

        entropy_estimates = []
        count_values = np.zeros(len(epsilon_range))

        for i, eps in enumerate(epsilon_range):
            # Count ε-distinguishable orbit segments of various lengths
            growth_rates = []
            for seg_len in range(2, min(max_orbit_length, T // 2)):
                n_segments = T - seg_len + 1
                if n_segments < 2:
                    break
                # Build orbit segments
                segments = np.array([
                    trajectory[j: j + seg_len].ravel()
                    for j in range(0, n_segments, max(1, n_segments // 200))
                ])
                # Count ε-separated segments (greedy covering)
                n_distinct = self._count_separated(segments, eps)
                if n_distinct > 1:
                    growth_rates.append(np.log(n_distinct) / seg_len)

            count_values[i] = len(growth_rates)
            if growth_rates:
                entropy_estimates.append(np.mean(growth_rates))
            else:
                entropy_estimates.append(0.0)

        entropy_estimates = np.array(entropy_estimates)

        # Estimate h_top as the limit (use median of fine-scale estimates)
        if len(entropy_estimates) > 0:
            # Use smaller epsilon values (more refined)
            n_fine = max(1, len(entropy_estimates) // 3)
            h_top = float(np.median(entropy_estimates[:n_fine]))
        else:
            h_top = 0.0

        # Correlation dimension from ε-scaling
        corr_dim = self._correlation_dimension(trajectory, epsilon_range)

        return TopologicalEntropyResult(
            entropy=max(0.0, h_top),
            epsilon_values=epsilon_range,
            count_values=count_values,
            correlation_dimension=corr_dim,
        )

    @staticmethod
    def _count_separated(segments: np.ndarray, epsilon: float) -> int:
        """Count ε-separated segments via greedy covering."""
        n = len(segments)
        if n == 0:
            return 0

        used = np.zeros(n, dtype=bool)
        count = 0

        for i in range(n):
            if used[i]:
                continue
            count += 1
            # Mark all segments within ε of segment i
            dists = np.linalg.norm(segments[i:] - segments[i], axis=1)
            used[i:][dists < epsilon] = True

        return count

    @staticmethod
    def _correlation_dimension(trajectory: np.ndarray, epsilons: np.ndarray) -> float:
        """Estimate correlation dimension from Grassberger-Procaccia algorithm."""
        T = len(trajectory)
        n_sample = min(T, 500)
        indices = np.random.choice(T, n_sample, replace=False)
        points = trajectory[indices]

        # Pairwise distances
        dists = []
        for i in range(n_sample):
            d = np.linalg.norm(points[i + 1:] - points[i], axis=1)
            dists.extend(d.tolist())
        dists = np.array(dists)

        if len(dists) == 0:
            return 0.0

        # Correlation sum C(ε) = fraction of pairs within ε
        log_eps = []
        log_c = []
        n_pairs = len(dists)

        for eps in epsilons:
            c = np.sum(dists < eps) / n_pairs
            if c > 0:
                log_eps.append(np.log(eps))
                log_c.append(np.log(c))

        if len(log_eps) < 3:
            return 0.0

        # Linear regression on log-log plot
        log_eps = np.array(log_eps)
        log_c = np.array(log_c)
        A = np.column_stack([log_eps, np.ones(len(log_eps))])
        try:
            slope = np.linalg.lstsq(A, log_c, rcond=None)[0][0]
        except np.linalg.LinAlgError:
            slope = 0.0

        return max(0.0, float(slope))


# ============================================================================
# FULL LYAPUNOV SPECTRUM
# ============================================================================

class LyapunovSpectrum:
    """Full Lyapunov spectrum computation via QR decomposition.

    Extends single Lyapunov exponent to the full spectrum
    {λ₁ ≥ λ₂ ≥ ... ≥ λ_d} by tracking d orthogonal perturbation
    vectors and re-orthogonalizing via QR at each step.

    The sum of positive exponents gives the Kolmogorov-Sinai entropy
    (Pesin's formula). The Kaplan-Yorke dimension indicates the
    information dimension of the attractor.

    Example:
        >>> ls = LyapunovSpectrum()
        >>> # Lorenz system trajectory
        >>> traj = simulate_lorenz(10000, dt=0.01)
        >>> result = ls.full_spectrum(traj, dt=0.01)
        >>> print(f"KS entropy: {result.kolmogorov_sinai_entropy:.4f}")
    """

    def full_spectrum(
        self,
        trajectory: np.ndarray,
        dt: float = 1.0,
        jacobian_fn: Callable | None = None,
        n_transient: int = 100,
        reorth_steps: int = 1,
    ) -> LyapunovSpectrumResult:
        """Compute full Lyapunov spectrum from trajectory or Jacobian.

        If jacobian_fn is provided, uses it directly. Otherwise,
        estimates the Jacobian from trajectory differences.

        Args:
            trajectory: Time series data (T, d).
            dt: Time step.
            jacobian_fn: Optional function J(x) -> (d, d) Jacobian matrix.
            n_transient: Steps to discard for transient dynamics.
            reorth_steps: Steps between QR re-orthogonalization.

        Returns:
            LyapunovSpectrumResult with full spectrum.
        """
        trajectory = np.atleast_2d(trajectory)
        if trajectory.shape[0] < trajectory.shape[1]:
            trajectory = trajectory.T
        T, d = trajectory.shape

        # Initialize orthonormal perturbation matrix
        Q = np.eye(d)
        log_stretches = np.zeros(d)
        n_steps = 0
        history = []

        start = max(n_transient, 1)

        for t in range(start, T - 1):
            # Get Jacobian at current point
            if jacobian_fn is not None:
                J = jacobian_fn(trajectory[t])
            else:
                J = self._estimate_jacobian(trajectory, t)

            # Propagate perturbations
            Q = J @ Q

            if (t - start + 1) % reorth_steps == 0:
                # QR decomposition for re-orthogonalization
                Q, R = np.linalg.qr(Q)
                # Accumulate stretching from diagonal of R
                diag_r = np.abs(np.diag(R))
                diag_r = np.maximum(diag_r, 1e-300)
                log_stretches += np.log(diag_r)
                n_steps += 1

                if n_steps > 0:
                    current = log_stretches / (n_steps * reorth_steps * dt)
                    history.append(current.copy())

        if n_steps == 0:
            return LyapunovSpectrumResult(
                exponents=np.zeros(d),
                kaplan_yorke_dimension=0.0,
                kolmogorov_sinai_entropy=0.0,
                sum_exponents=0.0,
                convergence_history=np.zeros((1, d)),
            )

        exponents = log_stretches / (n_steps * reorth_steps * dt)
        exponents = np.sort(exponents)[::-1]

        # Kaplan-Yorke dimension
        ky_dim = self._kaplan_yorke_dimension(exponents)

        # Kolmogorov-Sinai entropy (Pesin formula)
        ks_entropy = float(np.sum(exponents[exponents > 0]))

        convergence = np.array(history) if history else np.zeros((1, d))

        return LyapunovSpectrumResult(
            exponents=exponents,
            kaplan_yorke_dimension=ky_dim,
            kolmogorov_sinai_entropy=ks_entropy,
            sum_exponents=float(np.sum(exponents)),
            convergence_history=convergence,
        )

    @staticmethod
    def _estimate_jacobian(trajectory: np.ndarray, t: int) -> np.ndarray:
        """Estimate Jacobian from trajectory differences.

        Uses J ≈ Δx_{t+1} / Δx_t via nearby trajectory points.
        Falls back to identity perturbation if insufficient data.
        """
        T, d = trajectory.shape

        # Simple finite difference estimate
        # J ≈ (x_{t+1} - x_t) as a linear map approximation
        if t + 1 >= T:
            return np.eye(d)

        dx = trajectory[t + 1] - trajectory[t]

        # Build approximate Jacobian from multiple nearby pairs
        # Find points near trajectory[t]
        dists = np.linalg.norm(trajectory[:T-1] - trajectory[t], axis=1)
        dists[t] = np.inf  # Exclude self

        k = min(d + 1, T - 1)
        nearest = np.argsort(dists)[:k]

        if len(nearest) < d:
            # Not enough neighbors — return identity perturbation
            J = np.eye(d)
            if np.linalg.norm(dx) > 0:
                J += 0.01 * np.outer(dx, dx) / (np.linalg.norm(dx) ** 2)
            return J

        # Build linear system: dx_i ≈ J @ (x_i - x_t) for neighbors
        X = trajectory[nearest] - trajectory[t]
        Y = trajectory[nearest + 1] - trajectory[t + 1]

        try:
            J = np.linalg.lstsq(X, Y, rcond=None)[0].T
        except np.linalg.LinAlgError:
            J = np.eye(d)

        return J

    @staticmethod
    def _kaplan_yorke_dimension(exponents: np.ndarray) -> float:
        """Compute Kaplan-Yorke (Lyapunov) dimension.

        D_KY = j + (Σ_{i=1}^{j} λ_i) / |λ_{j+1}|

        where j is the largest index such that Σ_{i=1}^{j} λ_i ≥ 0.
        """
        d = len(exponents)
        cumsum = np.cumsum(exponents)

        # Find j where cumsum transitions from positive to negative
        j = 0
        for i in range(d):
            if cumsum[i] >= 0:
                j = i
            else:
                break

        if j >= d - 1:
            return float(d)

        if abs(exponents[j + 1]) < 1e-300:
            return float(j + 1)

        return float(j + 1 + cumsum[j] / abs(exponents[j + 1]))

    def from_map(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        jacobian_fn: Callable[[np.ndarray], np.ndarray],
        n_iter: int = 10000,
        n_transient: int = 1000,
    ) -> LyapunovSpectrumResult:
        """Compute spectrum directly from a map and its Jacobian.

        More accurate than trajectory-based estimation when the
        Jacobian is known analytically.

        Args:
            f: Map x_{n+1} = f(x_n).
            x0: Initial condition.
            jacobian_fn: Jacobian Df(x) as function of x.
            n_iter: Total iterations.
            n_transient: Transient iterations to discard.

        Returns:
            LyapunovSpectrumResult.
        """
        x = np.asarray(x0, dtype=float)
        d = len(x)
        Q = np.eye(d)
        log_stretches = np.zeros(d)
        n_steps = 0
        history = []

        for t in range(n_iter):
            J = jacobian_fn(x)
            x = f(x)
            Q = J @ Q

            # QR re-orthogonalization
            Q, R = np.linalg.qr(Q)
            diag_r = np.abs(np.diag(R))
            diag_r = np.maximum(diag_r, 1e-300)

            if t >= n_transient:
                log_stretches += np.log(diag_r)
                n_steps += 1
                if n_steps % 100 == 0:
                    history.append((log_stretches / n_steps).copy())

        if n_steps == 0:
            return LyapunovSpectrumResult(
                exponents=np.zeros(d), kaplan_yorke_dimension=0.0,
                kolmogorov_sinai_entropy=0.0, sum_exponents=0.0,
                convergence_history=np.zeros((1, d)),
            )

        exponents = np.sort(log_stretches / n_steps)[::-1]
        ky_dim = self._kaplan_yorke_dimension(exponents)
        ks_entropy = float(np.sum(exponents[exponents > 0]))

        convergence = np.array(history) if history else np.zeros((1, d))

        return LyapunovSpectrumResult(
            exponents=exponents,
            kaplan_yorke_dimension=ky_dim,
            kolmogorov_sinai_entropy=ks_entropy,
            sum_exponents=float(np.sum(exponents)),
            convergence_history=convergence,
        )
