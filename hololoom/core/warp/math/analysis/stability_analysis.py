"""
Stability Analysis — Lyapunov Exponents, Fixed Points, Linear Stability
=======================================================================

Dynamical systems stability analysis tools.

Core Concepts:
- Lyapunov Exponents: Measure sensitivity to initial conditions
- Fixed Point Analysis: Locate and classify equilibria
- Linear Stability: Eigenvalue-based stability of linearized systems
- Routh-Hurwitz: Polynomial stability criterion without computing roots

Mathematical Foundation:
Lyapunov exponent: λ = lim_{t→∞} (1/t) ln(|δx(t)| / |δx(0)|)
Linear stability: classify via eigenvalues of Jacobian at fixed point
Routh-Hurwitz: all roots of p(s) have Re(s) < 0 iff all Hurwitz determinants > 0

Applications to Warp Space:
- Convergence detection in iterative weaving cycles
- Stability of embedding dynamics under perturbation
- Chaos detection in multi-agent Thompson Sampling
- Fixed point analysis for scoring term equilibria

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy.linalg import qr
    from scipy.optimize import fsolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class LyapunovResult:
    """Result of Lyapunov exponent computation.

    Attributes:
        exponents: Array of Lyapunov exponents, sorted descending.
        max_exponent: Largest Lyapunov exponent (λ_max).
        is_chaotic: True if λ_max > 0 (positive largest exponent).
        convergence_time: Number of steps until exponents stabilized, or None.
    """
    exponents: np.ndarray
    max_exponent: float
    is_chaotic: bool
    convergence_time: float | None = None

    @property
    def lyapunov_dimension(self) -> float:
        """Kaplan-Yorke dimension estimate.

        D_KY = j + (1/|λ_{j+1}|) * Σ_{i=1}^{j} λ_i
        where j is the largest index such that Σ_{i=1}^{j} λ_i >= 0.
        """
        sorted_exp = np.sort(self.exponents)[::-1]
        cumsum = np.cumsum(sorted_exp)
        j_indices = np.where(cumsum >= 0)[0]
        if len(j_indices) == 0:
            return 0.0
        j = j_indices[-1]
        if j + 1 >= len(sorted_exp):
            return float(len(sorted_exp))
        if abs(sorted_exp[j + 1]) < 1e-15:
            return float(j + 1)
        return float(j + 1) + cumsum[j] / abs(sorted_exp[j + 1])

    @property
    def entropy_rate(self) -> float:
        """Kolmogorov-Sinai entropy estimate (sum of positive exponents)."""
        return float(np.sum(self.exponents[self.exponents > 0]))


@dataclass
class FixedPoint:
    """A fixed point of a dynamical system with stability classification.

    Attributes:
        location: Coordinates of the fixed point.
        jacobian: Jacobian matrix evaluated at the fixed point.
        stability: Classification string.
    """
    location: np.ndarray
    jacobian: np.ndarray
    stability: str  # stable_node, unstable_node, saddle, center, stable_spiral, unstable_spiral

    @property
    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the Jacobian at this fixed point."""
        return np.linalg.eigvals(self.jacobian)

    @property
    def is_stable(self) -> bool:
        """True if the fixed point is stable (all eigenvalues have negative real part)."""
        return self.stability in ("stable_node", "stable_spiral")

    @property
    def dimension(self) -> int:
        """Dimension of the phase space."""
        return len(self.location)


# ============================================================================
# LYAPUNOV EXPONENTS
# ============================================================================

class LyapunovExponents:
    """
    Compute Lyapunov exponents for dynamical systems.

    Lyapunov exponents quantify the rate of separation of infinitesimally
    close trajectories. A positive largest exponent indicates chaos.
    """

    @staticmethod
    def compute_from_trajectory(
        trajectory: np.ndarray,
        dt: float = 1.0,
        n_neighbors: int = 1,
        renorm_steps: int = 10
    ) -> LyapunovResult:
        """
        Estimate Lyapunov exponents from a time series using Wolf's algorithm.

        Tracks the growth of perturbation vectors along the trajectory,
        periodically renormalizing to prevent overflow. The exponents are
        the time-averaged logarithmic growth rates.

        Args:
            trajectory: Shape (T, d) state-space trajectory.
            dt: Time step between observations.
            n_neighbors: Not used directly; kept for API compatibility.
            renorm_steps: Renormalize perturbation vectors every this many steps.

        Returns:
            LyapunovResult with estimated exponents.
        """
        trajectory = np.asarray(trajectory, dtype=np.float64)
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(-1, 1)

        T, d = trajectory.shape
        if T < 2 * d + 1:
            logger.warning("Trajectory too short for reliable Lyapunov estimation")

        # Initialize perturbation matrix (orthonormal)
        Q = np.eye(d, dtype=np.float64)
        exponent_sums = np.zeros(d, dtype=np.float64)
        n_renorm = 0

        # Estimate local Jacobians via finite differences along trajectory
        for t in range(T - 1):
            # Approximate local Jacobian from trajectory differences
            if t + d < T:
                J = LyapunovExponents._local_jacobian(trajectory, t, dt)
            else:
                # Use last valid Jacobian estimate
                J = LyapunovExponents._local_jacobian(trajectory, max(0, T - d - 2), dt)

            # Evolve perturbation vectors
            Q = J @ Q

            # Periodic renormalization via QR decomposition
            if (t + 1) % renorm_steps == 0:
                Q_new, R = np.linalg.qr(Q)
                # Accumulate log of diagonal elements (growth rates)
                diag = np.abs(np.diag(R))
                diag = np.where(diag > 1e-300, diag, 1e-300)  # Prevent log(0)
                exponent_sums += np.log(diag)
                n_renorm += 1
                Q = Q_new

        # Final renormalization if needed
        if n_renorm == 0:
            Q_new, R = np.linalg.qr(Q)
            diag = np.abs(np.diag(R))
            diag = np.where(diag > 1e-300, diag, 1e-300)
            exponent_sums += np.log(diag)
            n_renorm = 1

        # Time-average the exponents
        total_time = n_renorm * renorm_steps * dt
        if total_time < 1e-15:
            total_time = dt
        exponents = exponent_sums / total_time

        # Sort descending
        exponents = np.sort(exponents)[::-1]

        max_exp = float(exponents[0])
        is_chaotic = max_exp > 0

        # Estimate convergence: check if exponents stabilized over last quarter
        convergence_time = None
        if T > 40:
            quarter = T // 4
            mid_exp = LyapunovExponents._partial_exponents(
                trajectory[:3 * quarter], dt, renorm_steps
            )
            full_exp = exponents
            if len(mid_exp) == len(full_exp):
                rel_change = np.max(np.abs(full_exp - mid_exp) / (np.abs(full_exp) + 1e-10))
                if rel_change < 0.1:
                    convergence_time = float(3 * quarter * dt)

        return LyapunovResult(
            exponents=exponents,
            max_exponent=max_exp,
            is_chaotic=is_chaotic,
            convergence_time=convergence_time,
        )

    @staticmethod
    def maximal_exponent(
        trajectory: np.ndarray,
        dt: float = 1.0,
        renorm_steps: int = 10
    ) -> float:
        """
        Compute only the largest Lyapunov exponent.

        More efficient than computing the full spectrum when only
        chaos detection is needed.

        Args:
            trajectory: Shape (T, d) state-space trajectory.
            dt: Time step between observations.
            renorm_steps: Renormalize every this many steps.

        Returns:
            Largest Lyapunov exponent (float).
        """
        trajectory = np.asarray(trajectory, dtype=np.float64)
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(-1, 1)

        T, d = trajectory.shape

        # Track a single perturbation vector
        v = np.random.randn(d)
        v = v / np.linalg.norm(v)
        log_sum = 0.0
        n_renorm = 0

        for t in range(T - 1):
            J = LyapunovExponents._local_jacobian(trajectory, t, dt)
            v = J @ v

            if (t + 1) % renorm_steps == 0:
                norm = np.linalg.norm(v)
                if norm > 1e-300:
                    log_sum += np.log(norm)
                    v = v / norm
                n_renorm += 1

        if n_renorm == 0:
            norm = np.linalg.norm(v)
            if norm > 1e-300:
                log_sum = np.log(norm)
            n_renorm = 1

        total_time = n_renorm * renorm_steps * dt
        if total_time < 1e-15:
            total_time = dt

        return log_sum / total_time

    @staticmethod
    def compute_from_jacobian(
        jacobian_fn: Callable[[np.ndarray], np.ndarray],
        trajectory: np.ndarray,
        dt: float = 1.0,
        renorm_steps: int = 10
    ) -> LyapunovResult:
        """
        Compute Lyapunov exponents given an analytical Jacobian function.

        More accurate than trajectory-based estimation since the Jacobian
        is exact rather than approximated from finite differences.

        Args:
            jacobian_fn: Function mapping state x to Jacobian matrix J(x).
            trajectory: Shape (T, d) state-space trajectory.
            dt: Time step.
            renorm_steps: Renormalize every this many steps.

        Returns:
            LyapunovResult with computed exponents.
        """
        trajectory = np.asarray(trajectory, dtype=np.float64)
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(-1, 1)

        T, d = trajectory.shape
        Q = np.eye(d, dtype=np.float64)
        exponent_sums = np.zeros(d, dtype=np.float64)
        n_renorm = 0

        for t in range(T):
            J = jacobian_fn(trajectory[t])
            # Linearized flow: approximate as I + J*dt for small dt
            M = np.eye(d) + J * dt
            Q = M @ Q

            if (t + 1) % renorm_steps == 0:
                Q_new, R = np.linalg.qr(Q)
                diag = np.abs(np.diag(R))
                diag = np.where(diag > 1e-300, diag, 1e-300)
                exponent_sums += np.log(diag)
                n_renorm += 1
                Q = Q_new

        if n_renorm == 0:
            Q_new, R = np.linalg.qr(Q)
            diag = np.abs(np.diag(R))
            diag = np.where(diag > 1e-300, diag, 1e-300)
            exponent_sums += np.log(diag)
            n_renorm = 1

        total_time = n_renorm * renorm_steps * dt
        if total_time < 1e-15:
            total_time = dt

        exponents = exponent_sums / total_time
        exponents = np.sort(exponents)[::-1]

        return LyapunovResult(
            exponents=exponents,
            max_exponent=float(exponents[0]),
            is_chaotic=bool(exponents[0] > 0),
            convergence_time=None,
        )

    @staticmethod
    def _local_jacobian(
        trajectory: np.ndarray,
        t: int,
        dt: float
    ) -> np.ndarray:
        """
        Estimate local Jacobian from trajectory via finite differences.

        Uses the displacement between successive points as an approximation
        of the linearized flow map.

        Args:
            trajectory: Full trajectory array.
            t: Current time index.
            dt: Time step.

        Returns:
            Estimated Jacobian matrix (d x d).
        """
        T, d = trajectory.shape

        # Simple forward difference: dx/dt ≈ (x_{t+1} - x_t) / dt
        # For Jacobian, we need d independent directions.
        # Use nearby trajectory points as perturbed states.

        if t + 1 >= T:
            return np.eye(d)

        # Base displacement
        dx = trajectory[t + 1] - trajectory[t]

        # Build Jacobian from trajectory structure
        # For 1D, J = dx/dt / x_t (scalar)
        # For higher D, use embedding theorem approach
        if d == 1:
            x_val = trajectory[t, 0]
            if abs(x_val) < 1e-15:
                return np.array([[dx[0] / dt]])
            return np.array([[dx[0] / (x_val * dt)]])

        # Construct approximate Jacobian from delay embedding
        J = np.eye(d, dtype=np.float64)

        # Collect nearby trajectory segments for regression
        window = min(d + 1, T - t)
        if window < 2:
            return J

        # Use least squares on local neighborhood
        X_local = np.zeros((window - 1, d))
        Y_local = np.zeros((window - 1, d))
        for k in range(window - 1):
            idx = t + k
            if idx + 1 < T:
                X_local[k] = trajectory[idx]
                Y_local[k] = (trajectory[idx + 1] - trajectory[idx]) / dt

        # Center the data
        x_mean = X_local.mean(axis=0)
        X_centered = X_local - x_mean

        # Solve for J: Y ≈ J @ X_centered.T
        gram = X_centered.T @ X_centered
        reg = 1e-10 * np.eye(d)
        try:
            J = (Y_local.T @ X_centered) @ np.linalg.inv(gram + reg)
        except np.linalg.LinAlgError:
            J = np.eye(d)

        return J

    @staticmethod
    def _partial_exponents(
        trajectory: np.ndarray,
        dt: float,
        renorm_steps: int
    ) -> np.ndarray:
        """Compute exponents on a partial trajectory (for convergence check)."""
        result = LyapunovExponents.compute_from_trajectory(
            trajectory, dt=dt, renorm_steps=renorm_steps
        )
        return result.exponents


# ============================================================================
# LINEAR STABILITY ANALYSIS
# ============================================================================

class LinearStability:
    """
    Linear stability analysis via eigenvalue decomposition.

    Given a Jacobian matrix at a fixed point, determine the local
    stability properties of the linearized system dx/dt = J x.
    """

    @staticmethod
    def eigenvalue_analysis(jacobian: np.ndarray) -> dict[str, Any]:
        """
        Full eigenvalue-based stability analysis of a Jacobian matrix.

        Args:
            jacobian: Square Jacobian matrix (d x d).

        Returns:
            Dictionary with:
                eigenvalues: Complex eigenvalues.
                eigenvectors: Corresponding eigenvectors.
                stability: Classification string.
                stable_dims: Number of stable dimensions.
                unstable_dims: Number of unstable dimensions.
                center_dims: Number of center dimensions (Re(λ) ≈ 0).
        """
        jacobian = np.asarray(jacobian, dtype=np.float64)
        d = jacobian.shape[0]
        if jacobian.shape != (d, d):
            raise ValueError(f"Jacobian must be square, got shape {jacobian.shape}")

        eigenvalues, eigenvectors = np.linalg.eig(jacobian)

        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)

        tol = 1e-10
        stable_dims = int(np.sum(real_parts < -tol))
        unstable_dims = int(np.sum(real_parts > tol))
        center_dims = int(np.sum(np.abs(real_parts) <= tol))
        has_oscillation = bool(np.any(np.abs(imag_parts) > tol))

        stability = LinearStability._classify_eigenvalues(
            real_parts, imag_parts, d, tol
        )

        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "stability": stability,
            "stable_dims": stable_dims,
            "unstable_dims": unstable_dims,
            "center_dims": center_dims,
            "has_oscillation": has_oscillation,
            "spectral_radius": float(np.max(np.abs(eigenvalues))),
            "spectral_abscissa": float(np.max(real_parts)),
        }

    @staticmethod
    def routh_hurwitz(coefficients: np.ndarray) -> bool:
        """
        Routh-Hurwitz stability criterion.

        Tests whether all roots of a polynomial have negative real parts,
        without actually computing the roots. The polynomial is:
            a_0 s^n + a_1 s^{n-1} + ... + a_n = 0

        For stability, all elements in the first column of the Routh array
        must have the same sign.

        Args:
            coefficients: Polynomial coefficients [a_0, a_1, ..., a_n],
                         highest degree first.

        Returns:
            True if all roots have negative real part (stable).
        """
        coefficients = np.asarray(coefficients, dtype=np.float64)
        n = len(coefficients) - 1  # Degree of polynomial

        if n < 1:
            return True

        # All coefficients must have same sign (necessary condition)
        if coefficients[0] < 0:
            coefficients = -coefficients

        if np.any(coefficients <= 0):
            # Zero or negative coefficients → at least one unstable root
            # (for a polynomial with all negative-real-part roots, all
            #  coefficients must be positive when leading coeff is positive)
            return False

        # Build Routh array
        # First two rows from polynomial coefficients
        rows = n + 1
        cols = (n + 2) // 2
        routh = np.zeros((rows, cols), dtype=np.float64)

        # Fill first two rows
        for i in range(cols):
            idx = 2 * i
            if idx < len(coefficients):
                routh[0, i] = coefficients[idx]
            idx = 2 * i + 1
            if idx < len(coefficients):
                routh[1, i] = coefficients[idx]

        # Fill remaining rows
        for i in range(2, rows):
            for j in range(cols - 1):
                if abs(routh[i - 1, 0]) < 1e-15:
                    # Zero in pivot position — use epsilon replacement
                    routh[i - 1, 0] = 1e-10
                routh[i, j] = (
                    routh[i - 1, 0] * routh[i - 2, j + 1]
                    - routh[i - 2, 0] * routh[i - 1, j + 1]
                ) / routh[i - 1, 0]

        # Check first column: all must be positive (same sign as a_0)
        first_col = routh[:, 0]
        return bool(np.all(first_col > 0))

    @staticmethod
    def stability_margin(jacobian: np.ndarray) -> float:
        """
        Compute the stability margin: distance from the eigenvalues
        to the imaginary axis.

        A positive margin means all eigenvalues have negative real parts
        (stable). The magnitude indicates robustness — how much the
        system can be perturbed before losing stability.

        Args:
            jacobian: Square Jacobian matrix.

        Returns:
            Stability margin (negative of spectral abscissa).
            Positive = stable, negative = unstable.
        """
        jacobian = np.asarray(jacobian, dtype=np.float64)
        eigenvalues = np.linalg.eigvals(jacobian)
        spectral_abscissa = float(np.max(np.real(eigenvalues)))
        return -spectral_abscissa

    @staticmethod
    def _classify_eigenvalues(
        real_parts: np.ndarray,
        imag_parts: np.ndarray,
        d: int,
        tol: float = 1e-10
    ) -> str:
        """
        Classify a fixed point based on eigenvalue structure.

        2D classification:
            stable_node: all Re(λ) < 0, no imaginary parts
            unstable_node: all Re(λ) > 0, no imaginary parts
            saddle: mixed signs of Re(λ)
            center: all Re(λ) ≈ 0 with imaginary parts
            stable_spiral: all Re(λ) < 0 with imaginary parts
            unstable_spiral: all Re(λ) > 0 with imaginary parts

        Higher dimensions use analogous classification.
        """
        has_imag = bool(np.any(np.abs(imag_parts) > tol))
        all_neg = bool(np.all(real_parts < -tol))
        all_pos = bool(np.all(real_parts > tol))
        all_zero = bool(np.all(np.abs(real_parts) <= tol))
        mixed = not all_neg and not all_pos and not all_zero

        if mixed:
            return "saddle"
        if all_zero and has_imag:
            return "center"
        if all_neg and has_imag:
            return "stable_spiral"
        if all_pos and has_imag:
            return "unstable_spiral"
        if all_neg:
            return "stable_node"
        if all_pos:
            return "unstable_node"
        if all_zero:
            return "center"

        return "saddle"


# ============================================================================
# FIXED POINT FINDER
# ============================================================================

class FixedPointFinder:
    """
    Locate and classify fixed points of dynamical systems.

    A fixed point x* satisfies f(x*) = 0 for a system dx/dt = f(x).
    Uses Newton's method (or scipy.optimize.fsolve when available)
    to find roots, then classifies via eigenvalue analysis.
    """

    @staticmethod
    def find(
        f: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        method: str = "newton",
        tol: float = 1e-8,
        max_iter: int = 100,
        jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> FixedPoint:
        """
        Find a single fixed point starting from initial guess x0.

        Args:
            f: System function f(x) — fixed point is where f(x*) = 0.
            x0: Initial guess (1D array).
            method: 'newton' for Newton-Raphson, 'scipy' for fsolve.
            tol: Convergence tolerance.
            max_iter: Maximum iterations.
            jacobian_fn: Optional analytical Jacobian. If None, uses
                        finite differences.

        Returns:
            FixedPoint with location, Jacobian, and stability classification.
        """
        x0 = np.asarray(x0, dtype=np.float64).ravel()
        d = len(x0)

        if method == "scipy" and HAS_SCIPY:
            x_star, info, ier, msg = fsolve(f, x0, full_output=True)
            if ier != 1:
                logger.warning(f"fsolve did not converge: {msg}")
            x_star = np.asarray(x_star, dtype=np.float64)
        else:
            x_star = FixedPointFinder._newton_method(
                f, x0, tol, max_iter, jacobian_fn
            )

        # Compute Jacobian at fixed point
        if jacobian_fn is not None:
            J = jacobian_fn(x_star)
        else:
            J = FixedPointFinder._numerical_jacobian(f, x_star)

        stability = FixedPointFinder.classify(J)

        return FixedPoint(
            location=x_star,
            jacobian=J,
            stability=stability,
        )

    @staticmethod
    def classify(jacobian: np.ndarray) -> str:
        """
        Classify a fixed point based on the Jacobian eigenvalues.

        Args:
            jacobian: Jacobian matrix at the fixed point.

        Returns:
            Classification string: stable_node, unstable_node, saddle,
            center, stable_spiral, unstable_spiral.
        """
        analysis = LinearStability.eigenvalue_analysis(jacobian)
        return analysis["stability"]

    @staticmethod
    def find_all(
        f: Callable[[np.ndarray], np.ndarray],
        x_range: np.ndarray,
        n_initial: int = 20,
        tol: float = 1e-8,
        max_iter: int = 100,
        merge_tol: float = 1e-6,
        jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> list[FixedPoint]:
        """
        Find multiple fixed points by searching from many initial conditions.

        Generates initial guesses uniformly within x_range, runs Newton's
        method from each, then deduplicates nearby results.

        Args:
            f: System function f(x).
            x_range: Shape (d, 2) array with [min, max] per dimension.
            n_initial: Number of initial conditions to try.
            tol: Convergence tolerance for each Newton solve.
            max_iter: Maximum iterations per solve.
            merge_tol: Distance threshold for merging duplicate fixed points.
            jacobian_fn: Optional analytical Jacobian.

        Returns:
            List of unique FixedPoint objects found.
        """
        x_range = np.asarray(x_range, dtype=np.float64)
        if x_range.ndim == 1:
            x_range = x_range.reshape(1, 2)
        d = x_range.shape[0]

        # Generate initial conditions (uniform random within range)
        initial_points = np.random.uniform(
            x_range[:, 0], x_range[:, 1], size=(n_initial, d)
        )

        candidates: list[FixedPoint] = []

        for x0 in initial_points:
            try:
                fp = FixedPointFinder.find(
                    f, x0, method="newton", tol=tol,
                    max_iter=max_iter, jacobian_fn=jacobian_fn
                )
                # Verify it's actually a fixed point
                residual = np.linalg.norm(f(fp.location))
                if residual < tol * 100:
                    # Check if within x_range (reject escaped solutions)
                    in_range = np.all(fp.location >= x_range[:, 0] - 1.0) and \
                               np.all(fp.location <= x_range[:, 1] + 1.0)
                    if in_range:
                        candidates.append(fp)
            except (np.linalg.LinAlgError, ValueError, RuntimeError):
                continue

        # Deduplicate: merge fixed points that are closer than merge_tol
        unique: list[FixedPoint] = []
        for fp in candidates:
            is_duplicate = False
            for existing in unique:
                dist = np.linalg.norm(fp.location - existing.location)
                if dist < merge_tol:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(fp)

        return unique

    @staticmethod
    def _newton_method(
        f: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        tol: float,
        max_iter: int,
        jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        """
        Newton-Raphson iteration for finding roots of f(x) = 0.

        x_{n+1} = x_n - J(x_n)^{-1} f(x_n)

        Args:
            f: Function to find root of.
            x0: Initial guess.
            tol: Convergence tolerance.
            max_iter: Maximum iterations.
            jacobian_fn: Analytical Jacobian, or None for finite differences.

        Returns:
            Approximate root.
        """
        x = x0.copy()

        for i in range(max_iter):
            fx = np.asarray(f(x), dtype=np.float64).ravel()

            if np.linalg.norm(fx) < tol:
                return x

            if jacobian_fn is not None:
                J = jacobian_fn(x)
            else:
                J = FixedPointFinder._numerical_jacobian(f, x)

            try:
                # Solve J @ dx = -fx
                dx = np.linalg.solve(J, -fx)
            except np.linalg.LinAlgError:
                # Singular Jacobian — add regularization
                reg = 1e-8 * np.eye(len(x))
                dx = np.linalg.solve(J + reg, -fx)

            x = x + dx

            if np.linalg.norm(dx) < tol:
                return x

        logger.warning(f"Newton's method did not converge in {max_iter} iterations")
        return x

    @staticmethod
    def _numerical_jacobian(
        f: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        h: float = 1e-7,
    ) -> np.ndarray:
        """
        Compute Jacobian via central finite differences.

        J_{ij} = (f_i(x + h*e_j) - f_i(x - h*e_j)) / (2h)

        Args:
            f: Vector-valued function.
            x: Point at which to evaluate Jacobian.
            h: Step size for finite differences.

        Returns:
            Jacobian matrix (m x n) where m = dim(f), n = dim(x).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        n = len(x)
        f0 = np.asarray(f(x), dtype=np.float64).ravel()
        m = len(f0)

        J = np.zeros((m, n), dtype=np.float64)

        for j in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += h
            x_minus[j] -= h

            f_plus = np.asarray(f(x_plus), dtype=np.float64).ravel()
            f_minus = np.asarray(f(x_minus), dtype=np.float64).ravel()

            J[:, j] = (f_plus - f_minus) / (2.0 * h)

        return J


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def is_chaotic(trajectory: np.ndarray, dt: float = 1.0) -> bool:
    """Quick check: is the largest Lyapunov exponent positive?"""
    return LyapunovExponents.maximal_exponent(trajectory, dt) > 0


def stability_classification(jacobian: np.ndarray) -> str:
    """Classify stability from a Jacobian matrix."""
    return FixedPointFinder.classify(jacobian)


def find_equilibria(
    f: Callable[[np.ndarray], np.ndarray],
    x_range: np.ndarray,
    n_initial: int = 20,
) -> list[FixedPoint]:
    """Convenience wrapper for FixedPointFinder.find_all."""
    return FixedPointFinder.find_all(f, x_range, n_initial=n_initial)
