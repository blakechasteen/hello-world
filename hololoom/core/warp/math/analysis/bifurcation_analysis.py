"""
Bifurcation Analysis — Critical Transitions in Dynamical Systems
=================================================================

Detection and classification of qualitative changes in system behavior
as parameters vary continuously.

Core Concepts:
- Bifurcation: qualitative change in the number or stability of equilibria
  as a parameter crosses a critical value
- Saddle-node: two fixed points collide and annihilate (real eigenvalue → 0)
- Hopf: fixed point loses stability, limit cycle born (complex pair crosses axis)
- Pitchfork: symmetric branching (one → three fixed points)
- Transcritical: exchange of stability between two fixed points
- Critical slowing down: recovery time diverges near bifurcation

Mathematical Foundation:
For dx/dt = f(x, μ) where μ is the bifurcation parameter:
- Fixed points: f(x*, μ) = 0
- Stability: sign of ∂f/∂x at x*
- Bifurcation: ∂f/∂x = 0 (eigenvalue crosses zero)

Normal forms:
  Saddle-node:   dx/dt = μ - x²
  Transcritical: dx/dt = μx - x²
  Pitchfork:     dx/dt = μx - x³ (supercritical)
  Hopf:          dr/dt = μr - r³, dθ/dt = ω (polar coordinates)

Imports FixedPointFinder, LinearStability, FixedPoint from stability_analysis.py.

Applications to Warp Space:
- Detecting phase transitions in Thompson Sampling dynamics
- Mode switching in scoring term equilibria
- Convergence/divergence boundaries in weaving cycles
- Critical parameter identification for system tuning

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .stability_analysis import FixedPointFinder

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class BifurcationPoint:
    """A bifurcation in parameter space.

    Attributes:
        parameter_value: Critical value of the bifurcation parameter μ_c.
        type: Classification: 'saddle_node', 'hopf', 'pitchfork',
              'transcritical', or 'unknown'.
        eigenvalues_at: Eigenvalues of the Jacobian at the bifurcation point.
        fixed_point: Location of the fixed point at the bifurcation.
    """
    parameter_value: float
    type: str
    eigenvalues_at: np.ndarray
    fixed_point: np.ndarray


@dataclass
class BifurcationDiagram:
    """Bifurcation diagram: fixed points tracked over parameter range.

    Attributes:
        parameter_values: Array of parameter values swept, shape (n_points,).
        fixed_points: List of arrays, one per parameter value. Each array
                      contains the fixed point locations found at that μ.
        stability: List of lists of bools. stability[i][j] is True if
                   fixed_points[i][j] is stable.
    """
    parameter_values: np.ndarray
    fixed_points: list[np.ndarray]
    stability: list[list[bool]]


# ============================================================================
# BIFURCATION ANALYZER
# ============================================================================

class BifurcationAnalyzer:
    """
    Detect and classify bifurcations in parameterized dynamical systems.

    Given a system dx/dt = f(x, μ), sweeps the parameter μ across a range,
    tracks fixed points, and detects qualitative changes in stability.
    """

    @staticmethod
    def find_bifurcations(
        f_param: Callable[[np.ndarray, float], np.ndarray],
        param_range: tuple[float, float],
        x0: np.ndarray,
        n_points: int = 100,
        jacobian_fn: Callable[[np.ndarray, float], np.ndarray] | None = None
    ) -> list[BifurcationPoint]:
        """
        Find bifurcation points by sweeping parameter and detecting stability changes.

        For each parameter value, finds fixed points near x0 using Newton's method,
        evaluates the Jacobian eigenvalues, and flags transitions where eigenvalues
        cross the imaginary axis.

        Args:
            f_param: System function f(x, μ). Takes state x and parameter μ.
            param_range: (μ_min, μ_max) range to sweep.
            x0: Initial guess for fixed point search.
            n_points: Number of parameter values to sample.
            jacobian_fn: Optional analytical Jacobian J(x, μ). If None, uses
                        finite differences.

        Returns:
            List of BifurcationPoint objects found.
        """
        x0 = np.asarray(x0, dtype=np.float64).ravel()
        d = len(x0)
        mu_values = np.linspace(param_range[0], param_range[1], n_points)

        bifurcations: list[BifurcationPoint] = []
        prev_eigenvalues = None
        prev_fp = None
        x_guess = x0.copy()

        for i, mu in enumerate(mu_values):
            # Define f(x) at this parameter value
            def f_at_mu(x: np.ndarray) -> np.ndarray:
                return np.asarray(f_param(x, mu), dtype=np.float64).ravel()

            jac_at_mu = None
            if jacobian_fn is not None:
                def jac_at_mu(x: np.ndarray) -> np.ndarray:
                    return np.asarray(jacobian_fn(x, mu), dtype=np.float64)

            # Find fixed point
            try:
                fp = FixedPointFinder.find(f_at_mu, x_guess, jacobian_fn=jac_at_mu)

                # Verify residual
                residual = np.linalg.norm(f_at_mu(fp.location))
                if residual > 1e-4:
                    continue

                eigenvalues = fp.eigenvalues
                x_guess = fp.location.copy()  # Use as next initial guess

                # Detect bifurcation: eigenvalue crossing
                if prev_eigenvalues is not None and len(eigenvalues) == len(prev_eigenvalues):
                    bif_type = BifurcationAnalyzer.classify(prev_eigenvalues, eigenvalues)
                    if bif_type is not None:
                        # Interpolate parameter value
                        mu_bif = 0.5 * (mu_values[i - 1] + mu)
                        bifurcations.append(BifurcationPoint(
                            parameter_value=mu_bif,
                            type=bif_type,
                            eigenvalues_at=eigenvalues,
                            fixed_point=fp.location.copy(),
                        ))
                        logger.info(f"Bifurcation ({bif_type}) at μ={mu_bif:.6f}")

                prev_eigenvalues = eigenvalues
                prev_fp = fp

            except (np.linalg.LinAlgError, ValueError, RuntimeError):
                # Fixed point finder failed — may indicate bifurcation
                prev_eigenvalues = None
                continue

        return bifurcations

    @staticmethod
    def diagram_1d(
        f_param: Callable[[float, float], float],
        param_range: tuple[float, float],
        x_range: tuple[float, float] = (-5.0, 5.0),
        n_points: int = 200,
        n_x_search: int = 50
    ) -> BifurcationDiagram:
        """
        Compute a 1D bifurcation diagram.

        For each parameter value μ, finds all fixed points x* where
        f(x*, μ) = 0 within x_range, and determines their stability.

        Optimized for scalar systems dx/dt = f(x, μ).

        Args:
            f_param: Scalar system function f(x, μ).
            param_range: (μ_min, μ_max).
            x_range: (x_min, x_max) search range for fixed points.
            n_points: Number of parameter values.
            n_x_search: Number of initial x values to search from.

        Returns:
            BifurcationDiagram with tracked fixed points and stability.
        """
        mu_values = np.linspace(param_range[0], param_range[1], n_points)
        x_search = np.linspace(x_range[0], x_range[1], n_x_search)

        all_fps: list[np.ndarray] = []
        all_stability: list[list[bool]] = []

        for mu in mu_values:
            fps_at_mu: list[float] = []
            stab_at_mu: list[bool] = []

            # Evaluate f across x range to find sign changes (roots)
            f_values = np.array([f_param(x, mu) for x in x_search])

            # Find sign changes
            for k in range(len(f_values) - 1):
                if f_values[k] * f_values[k + 1] < 0:
                    # Bisection to find root
                    x_lo, x_hi = x_search[k], x_search[k + 1]
                    for _ in range(60):  # 60 bisection steps → ~1e-18 precision
                        x_mid = 0.5 * (x_lo + x_hi)
                        f_mid = f_param(x_mid, mu)
                        if abs(f_mid) < 1e-14:
                            break
                        if f_mid * f_param(x_lo, mu) < 0:
                            x_hi = x_mid
                        else:
                            x_lo = x_mid
                    x_star = 0.5 * (x_lo + x_hi)

                    # Check for duplicates
                    is_dup = any(abs(x_star - xp) < 1e-6 for xp in fps_at_mu)
                    if not is_dup:
                        fps_at_mu.append(x_star)
                        # Stability: df/dx < 0 at x*
                        h = 1e-7
                        dfdx = (f_param(x_star + h, mu) - f_param(x_star - h, mu)) / (2 * h)
                        stab_at_mu.append(dfdx < 0)

                # Also check for roots at near-zero values
                elif abs(f_values[k]) < 1e-10:
                    x_star = x_search[k]
                    is_dup = any(abs(x_star - xp) < 1e-6 for xp in fps_at_mu)
                    if not is_dup:
                        fps_at_mu.append(x_star)
                        h = 1e-7
                        dfdx = (f_param(x_star + h, mu) - f_param(x_star - h, mu)) / (2 * h)
                        stab_at_mu.append(dfdx < 0)

            all_fps.append(np.array(fps_at_mu))
            all_stability.append(stab_at_mu)

        return BifurcationDiagram(
            parameter_values=mu_values,
            fixed_points=all_fps,
            stability=all_stability,
        )

    @staticmethod
    def classify(
        eigenvalues_before: np.ndarray,
        eigenvalues_after: np.ndarray
    ) -> str | None:
        """
        Classify a bifurcation from eigenvalue change.

        Detects:
        - saddle_node: a real eigenvalue crosses zero
        - hopf: a complex conjugate pair crosses the imaginary axis
        - pitchfork: real eigenvalue crosses zero with symmetry (detected
          as saddle_node — disambiguation requires counting fixed points)
        - transcritical: real eigenvalue crosses zero (stability exchange;
          detected as saddle_node — disambiguation requires tracking two
          branches)

        Args:
            eigenvalues_before: Eigenvalues at μ - δ.
            eigenvalues_after: Eigenvalues at μ + δ.

        Returns:
            Bifurcation type string, or None if no bifurcation detected.
        """
        ev_before = np.asarray(eigenvalues_before, dtype=np.complex128)
        ev_after = np.asarray(eigenvalues_after, dtype=np.complex128)

        if len(ev_before) != len(ev_after):
            return None

        re_before = np.real(ev_before)
        re_after = np.real(ev_after)
        im_before = np.imag(ev_before)
        im_after = np.imag(ev_after)

        tol = 1e-6

        # Check for Hopf bifurcation: complex pair crosses imaginary axis
        # Look for eigenvalues with nonzero imaginary part where real part changes sign
        for i in range(len(ev_before)):
            if abs(im_before[i]) > tol or abs(im_after[i]) > tol:
                # Has imaginary component — check if real part changed sign
                if re_before[i] * re_after[i] < 0:
                    return "hopf"

        # Check for saddle-node / transcritical / pitchfork:
        # A real eigenvalue crosses zero
        for i in range(len(ev_before)):
            if abs(im_before[i]) < tol and abs(im_after[i]) < tol:
                # Both real
                if re_before[i] * re_after[i] < 0:
                    return "saddle_node"

        # Check for stability change without sign crossing (near-zero passage)
        all_stable_before = np.all(re_before < -tol)
        all_stable_after = np.all(re_after < -tol)
        if all_stable_before != all_stable_after:
            # Stability changed but we didn't catch a clean crossing
            # Could be a degenerate case
            has_imag = np.any(np.abs(im_after) > tol) or np.any(np.abs(im_before) > tol)
            if has_imag:
                return "hopf"
            return "saddle_node"

        return None

    @staticmethod
    def critical_slowing_down(
        f_param: Callable[[np.ndarray, float], np.ndarray],
        param: float,
        x_eq: np.ndarray,
        perturbation: float = 0.01,
        dt: float = 0.01,
        max_steps: int = 100000,
        recovery_tol: float = 0.01,
        jacobian_fn: Callable[[np.ndarray, float], np.ndarray] | None = None
    ) -> float:
        """
        Measure critical slowing down: recovery time after perturbation.

        Near a bifurcation point, the dominant eigenvalue approaches zero,
        causing the system to take longer to return to equilibrium after
        a perturbation. This recovery time diverges at the bifurcation.

        Method: perturb the equilibrium, simulate forward via Euler, and
        measure the time to return within recovery_tol of x_eq.

        Args:
            f_param: System function f(x, μ).
            param: Current parameter value μ.
            x_eq: Equilibrium point x*.
            perturbation: Magnitude of initial perturbation.
            dt: Simulation time step.
            max_steps: Maximum simulation steps before giving up.
            recovery_tol: Tolerance for "recovered" (distance < tol).
            jacobian_fn: Optional Jacobian for analytical estimate.

        Returns:
            Recovery time (float). Returns max_steps * dt if not recovered.
        """
        x_eq = np.asarray(x_eq, dtype=np.float64).ravel()

        # If Jacobian available, compute analytical recovery time
        if jacobian_fn is not None:
            J = np.asarray(jacobian_fn(x_eq, param), dtype=np.float64)
            eigenvalues = np.linalg.eigvals(J)
            real_parts = np.real(eigenvalues)
            # Recovery time ~ 1/|max real eigenvalue| (for stable system)
            max_real = np.max(real_parts)
            if max_real < -1e-15:
                return float(-1.0 / max_real)
            # Unstable or marginally stable
            return float(max_steps * dt)

        # Numerical simulation
        d = len(x_eq)
        # Random perturbation direction
        direction = np.random.randn(d)
        direction = direction / (np.linalg.norm(direction) + 1e-300)

        x = x_eq + perturbation * direction

        for step in range(max_steps):
            f_val = np.asarray(f_param(x, param), dtype=np.float64).ravel()
            x = x + dt * f_val

            dist = np.linalg.norm(x - x_eq)
            if dist < recovery_tol:
                return float(step * dt)

            # Divergence check
            if dist > 1e6 or np.any(np.isnan(x)):
                return float(max_steps * dt)

        return float(max_steps * dt)

    @staticmethod
    def continuation_1d(
        f_param: Callable[[float, float], float],
        x0: float,
        mu0: float,
        mu_end: float,
        n_steps: int = 500,
        ds: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Pseudo-arclength continuation for 1D systems.

        Tracks a branch of fixed points f(x, μ) = 0 by parameterizing
        along arclength instead of μ, allowing the curve to turn around
        fold points (saddle-node bifurcations).

        Args:
            f_param: Scalar function f(x, μ).
            x0: Starting fixed point.
            mu0: Starting parameter value.
            mu_end: Target parameter value (for direction).
            n_steps: Number of continuation steps.
            ds: Arclength step size.

        Returns:
            (x_branch, mu_branch) arrays of shape (n_steps,).
        """
        h = 1e-7
        x_branch = np.zeros(n_steps, dtype=np.float64)
        mu_branch = np.zeros(n_steps, dtype=np.float64)
        x_branch[0] = x0
        mu_branch[0] = mu0

        # Initial direction: tangent to curve f(x,μ)=0
        fx = (f_param(x0 + h, mu0) - f_param(x0 - h, mu0)) / (2 * h)
        fmu = (f_param(x0, mu0 + h) - f_param(x0, mu0 - h)) / (2 * h)

        # Tangent: fx*dx + fmu*dmu = 0 → (dx, dmu) ∝ (-fmu, fx)
        tangent_x = -fmu
        tangent_mu = fx
        norm_t = np.sqrt(tangent_x**2 + tangent_mu**2)
        if norm_t < 1e-15:
            tangent_x, tangent_mu = 0.0, 1.0
            norm_t = 1.0
        tangent_x /= norm_t
        tangent_mu /= norm_t

        # Ensure initial direction towards mu_end
        if tangent_mu * (mu_end - mu0) < 0:
            tangent_x = -tangent_x
            tangent_mu = -tangent_mu

        for k in range(1, n_steps):
            # Predictor: step along tangent
            x_pred = x_branch[k - 1] + ds * tangent_x
            mu_pred = mu_branch[k - 1] + ds * tangent_mu

            # Corrector: Newton on augmented system
            # F1 = f(x, μ) = 0
            # F2 = tangent_x*(x - x_pred_0) + tangent_mu*(μ - mu_pred_0) = 0 (arclength constraint)
            x_cur = x_pred
            mu_cur = mu_pred
            x_pred_0 = x_branch[k - 1] + ds * tangent_x
            mu_pred_0 = mu_branch[k - 1] + ds * tangent_mu

            for _ in range(20):
                F1 = f_param(x_cur, mu_cur)
                F2 = tangent_x * (x_cur - x_pred_0) + tangent_mu * (mu_cur - mu_pred_0)

                # Jacobian of augmented system
                fx_c = (f_param(x_cur + h, mu_cur) - f_param(x_cur - h, mu_cur)) / (2 * h)
                fmu_c = (f_param(x_cur, mu_cur + h) - f_param(x_cur, mu_cur - h)) / (2 * h)

                # [fx, fmu; tangent_x, tangent_mu] [dx; dmu] = -[F1; F2]
                det = fx_c * tangent_mu - fmu_c * tangent_x
                if abs(det) < 1e-15:
                    break

                dx = (-F1 * tangent_mu + fmu_c * F2) / det
                dmu = (-fx_c * F2 + F1 * tangent_x) / det

                x_cur += dx
                mu_cur += dmu

                if abs(F1) < 1e-12 and abs(F2) < 1e-12:
                    break

            x_branch[k] = x_cur
            mu_branch[k] = mu_cur

            # Update tangent
            fx_new = (f_param(x_cur + h, mu_cur) - f_param(x_cur - h, mu_cur)) / (2 * h)
            fmu_new = (f_param(x_cur, mu_cur + h) - f_param(x_cur, mu_cur - h)) / (2 * h)
            tangent_x = -fmu_new
            tangent_mu = fx_new
            norm_t = np.sqrt(tangent_x**2 + tangent_mu**2)
            if norm_t < 1e-15:
                break
            tangent_x /= norm_t
            tangent_mu /= norm_t

            # Maintain direction consistency
            prev_dx = x_branch[k] - x_branch[k - 1]
            prev_dmu = mu_branch[k] - mu_branch[k - 1]
            if tangent_x * prev_dx + tangent_mu * prev_dmu < 0:
                tangent_x = -tangent_x
                tangent_mu = -tangent_mu

        return x_branch, mu_branch


# ============================================================================
# HOPF BIFURCATION ANALYSIS
# ============================================================================

class HopfBifurcation:
    """
    Specialized analysis for Hopf bifurcations.

    A Hopf bifurcation occurs when a complex conjugate pair of eigenvalues
    crosses the imaginary axis. A stable fixed point becomes unstable and
    a limit cycle is born (supercritical) or an unstable limit cycle
    disappears (subcritical).
    """

    @staticmethod
    def detect(
        f_param: Callable[[np.ndarray, float], np.ndarray],
        jacobian_fn: Callable[[np.ndarray, float], np.ndarray],
        param_range: tuple[float, float],
        x0: np.ndarray,
        n_points: int = 200
    ) -> BifurcationPoint | None:
        """
        Detect a Hopf bifurcation by tracking complex eigenvalue pairs.

        Sweeps the parameter μ and monitors the real parts of complex
        eigenvalue pairs. A Hopf bifurcation is detected when a pair
        crosses from negative to positive real part.

        Args:
            f_param: System function f(x, μ).
            jacobian_fn: Analytical Jacobian J(x, μ).
            param_range: (μ_min, μ_max) to sweep.
            x0: Initial guess for fixed point.
            n_points: Number of parameter samples.

        Returns:
            BifurcationPoint if Hopf detected, else None.
        """
        x0 = np.asarray(x0, dtype=np.float64).ravel()
        mu_values = np.linspace(param_range[0], param_range[1], n_points)

        x_guess = x0.copy()
        prev_complex_real = None
        prev_mu = None

        for mu in mu_values:
            def f_at_mu(x):
                return np.asarray(f_param(x, mu), dtype=np.float64).ravel()

            def jac_at_mu(x):
                return np.asarray(jacobian_fn(x, mu), dtype=np.float64)

            try:
                fp = FixedPointFinder.find(f_at_mu, x_guess, jacobian_fn=jac_at_mu)
                residual = np.linalg.norm(f_at_mu(fp.location))
                if residual > 1e-4:
                    continue

                eigenvalues = fp.eigenvalues
                x_guess = fp.location.copy()

                # Find complex eigenvalues
                complex_mask = np.abs(np.imag(eigenvalues)) > 1e-8
                if np.any(complex_mask):
                    complex_eigs = eigenvalues[complex_mask]
                    # Take the one with positive imaginary part
                    pos_imag = complex_eigs[np.imag(complex_eigs) > 0]
                    if len(pos_imag) > 0:
                        current_real = np.real(pos_imag[0])

                        if prev_complex_real is not None:
                            if prev_complex_real * current_real < 0:
                                # Crossing detected — interpolate
                                frac = abs(prev_complex_real) / (abs(prev_complex_real) + abs(current_real))
                                mu_hopf = prev_mu + frac * (mu - prev_mu)

                                logger.info(f"Hopf bifurcation at μ={mu_hopf:.6f}")
                                return BifurcationPoint(
                                    parameter_value=mu_hopf,
                                    type="hopf",
                                    eigenvalues_at=eigenvalues,
                                    fixed_point=fp.location.copy(),
                                )

                        prev_complex_real = current_real
                        prev_mu = mu

            except (np.linalg.LinAlgError, ValueError, RuntimeError):
                continue

        return None

    @staticmethod
    def limit_cycle_amplitude(
        mu: float,
        param_critical: float,
        coefficient: float = 1.0
    ) -> float:
        """
        Estimate limit cycle amplitude from Hopf normal form.

        For a supercritical Hopf bifurcation with normal form:
          dr/dt = (μ - μ_c) r - a r³

        the limit cycle amplitude is:
          r* = √((μ - μ_c) / a)    for μ > μ_c (supercritical, a > 0)

        Args:
            mu: Current parameter value.
            param_critical: Critical parameter value μ_c.
            coefficient: Normal form coefficient a > 0 for supercritical.

        Returns:
            Limit cycle amplitude. Returns 0 if below critical.
        """
        if coefficient <= 0:
            raise ValueError("Coefficient must be positive for supercritical Hopf")

        delta = mu - param_critical
        if delta <= 0:
            return 0.0

        return float(np.sqrt(delta / coefficient))

    @staticmethod
    def frequency_at_bifurcation(
        jacobian_fn: Callable[[np.ndarray, float], np.ndarray],
        x_eq: np.ndarray,
        param_critical: float
    ) -> float:
        """
        Compute the oscillation frequency at the Hopf bifurcation.

        At the critical point, the eigenvalues are ±iω. The frequency
        ω determines the period of the emerging limit cycle: T = 2π/ω.

        Args:
            jacobian_fn: Analytical Jacobian J(x, μ).
            x_eq: Equilibrium point at the bifurcation.
            param_critical: Critical parameter value.

        Returns:
            Angular frequency ω.
        """
        x_eq = np.asarray(x_eq, dtype=np.float64).ravel()
        J = np.asarray(jacobian_fn(x_eq, param_critical), dtype=np.float64)
        eigenvalues = np.linalg.eigvals(J)

        # Find eigenvalue with largest imaginary part
        imag_parts = np.abs(np.imag(eigenvalues))
        idx = np.argmax(imag_parts)

        omega = float(imag_parts[idx])
        if omega < 1e-10:
            logger.warning("No oscillatory eigenvalue found at alleged Hopf point")

        return omega


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def saddle_node_normal_form(x: float, mu: float) -> float:
    """Normal form for saddle-node bifurcation: dx/dt = μ - x²."""
    return mu - x**2


def pitchfork_normal_form(x: float, mu: float, subcritical: bool = False) -> float:
    """Normal form for pitchfork bifurcation.

    Supercritical: dx/dt = μx - x³  (stable branches)
    Subcritical:   dx/dt = μx + x³  (unstable branches)
    """
    sign = 1.0 if subcritical else -1.0
    return mu * x + sign * x**3


def transcritical_normal_form(x: float, mu: float) -> float:
    """Normal form for transcritical bifurcation: dx/dt = μx - x²."""
    return mu * x - x**2


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'BifurcationPoint',
    'BifurcationDiagram',
    'BifurcationAnalyzer',
    'HopfBifurcation',
    'saddle_node_normal_form',
    'pitchfork_normal_form',
    'transcritical_normal_form',
]
