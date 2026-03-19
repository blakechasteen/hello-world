"""
Fokker-Planck Equation — Probability Density Evolution
=======================================================

Numerical solution of the Fokker-Planck equation (FPE) for the time
evolution of probability density functions in stochastic systems.

Core Concepts:
- FPE: ∂p/∂t = -∂/∂x [μ(x)p] + ½ ∂²/∂x² [D(x)p]
  where μ(x) is drift and D(x) is diffusion coefficient
- Dual to the SDE: dX = μ(X)dt + σ(X)dW where D = σ²
- Steady state: dp/dt = 0 → detailed balance or probability current = 0
- Kramers escape: thermal activation over potential barriers

Mathematical Foundation:
The FPE governs the probability density p(x,t) of a particle following
the Ito SDE dX = μ(X)dt + √(D(X))dW.

Steady state solution (reflecting boundaries, constant D):
  p_ss(x) ∝ exp(2 ∫ μ(x')/D dx')

Mean first passage time from x to target b:
  T(x) = 2 ∫_x^b (1/D(y) p_ss(y)) ∫_{-∞}^y p_ss(z) dz dy

Builds on stochastic_calculus.py (Brownian motion, SDEs).

Applications to Warp Space:
- Density evolution of embedding distributions under drift
- Convergence analysis of Thompson Sampling posteriors
- Noise-driven exploration dynamics in policy space
- Relaxation time estimation for scoring equilibria

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
class FPESolution:
    """Solution of the Fokker-Planck equation.

    Attributes:
        density: Probability density p(x,t), shape (nt, nx).
                 density[i, j] = p(x_grid[j], t_grid[i]).
        x_grid: Spatial grid points, shape (nx,).
        t_grid: Temporal grid points, shape (nt,).
        steady_state: Steady-state density p_ss(x), shape (nx,), or None.
    """
    density: np.ndarray
    x_grid: np.ndarray
    t_grid: np.ndarray
    steady_state: np.ndarray | None = None


# ============================================================================
# FOKKER-PLANCK SOLVER
# ============================================================================

class FokkerPlanckSolver:
    """
    Numerical solver for the 1D Fokker-Planck equation.

    ∂p/∂t = -∂/∂x [μ(x) p(x,t)] + ½ ∂²/∂x² [D(x) p(x,t)]

    Uses the Crank-Nicolson finite difference scheme for unconditional
    stability and second-order accuracy in both space and time.
    Reflecting (zero-flux) boundary conditions are applied by default.
    """

    @staticmethod
    def solve_1d(
        drift_fn: Callable[[float], float],
        diffusion_fn: Callable[[float], float],
        initial_density: np.ndarray,
        x_range: tuple[float, float],
        t_range: tuple[float, float],
        nx: int = 200,
        nt: int = 1000
    ) -> FPESolution:
        """
        Solve the 1D Fokker-Planck equation via Crank-Nicolson.

        The FPE is discretized as:
          (p^{n+1} - p^n)/dt = ½(L p^{n+1} + L p^n)

        where L is the spatial FPE operator. This yields a tridiagonal
        system at each time step, solved via Thomas algorithm.

        Boundary conditions: reflecting (zero probability flux at boundaries).

        Args:
            drift_fn: Drift coefficient μ(x). Callable: float -> float.
            diffusion_fn: Diffusion coefficient D(x) = σ²(x).
                          Callable: float -> float.
            initial_density: Initial density p(x, 0) on the spatial grid.
                            Shape (nx,). Will be normalized.
            x_range: Spatial domain (x_min, x_max).
            t_range: Time domain (t_start, t_end).
            nx: Number of spatial grid points.
            nt: Number of time steps.

        Returns:
            FPESolution with density evolution.
        """
        x_min, x_max = x_range
        t_start, t_end = t_range
        dx = (x_max - x_min) / (nx - 1)
        dt = (t_end - t_start) / (nt - 1)

        x_grid = np.linspace(x_min, x_max, nx)
        t_grid = np.linspace(t_start, t_end, nt)

        # Evaluate drift and diffusion on grid
        mu = np.array([drift_fn(x) for x in x_grid])
        D = np.array([diffusion_fn(x) for x in x_grid])
        D = np.maximum(D, 1e-15)  # Prevent zero diffusion

        # Initialize density (normalize to integrate to 1)
        p = np.asarray(initial_density, dtype=np.float64).copy()
        if len(p) != nx:
            # Interpolate to grid
            x_old = np.linspace(x_min, x_max, len(p))
            p = np.interp(x_grid, x_old, p)
        norm = np.trapz(p, x_grid)
        if norm > 0:
            p /= norm

        # Store density at each time step
        density = np.zeros((nt, nx), dtype=np.float64)
        density[0] = p

        # Build Crank-Nicolson coefficient matrices
        # FPE operator L: Lp = -d/dx[μp] + ½ d²/dx²[Dp]
        # Using central differences:
        #   -d/dx[μp]_j ≈ -(μ_{j+1}p_{j+1} - μ_{j-1}p_{j-1}) / (2dx)
        #   ½ d²/dx²[Dp]_j ≈ ½(D_{j+1}p_{j+1} - 2D_j p_j + D_{j-1}p_{j-1}) / dx²
        #
        # Combined: a_j p_{j-1} + b_j p_j + c_j p_{j+1}
        # where:
        #   a_j = μ_{j-1}/(2dx) + ½ D_{j-1}/dx²
        #   b_j = -D_j/dx²
        #   c_j = -μ_{j+1}/(2dx) + ½ D_{j+1}/dx²

        a = np.zeros(nx, dtype=np.float64)  # sub-diagonal
        b = np.zeros(nx, dtype=np.float64)  # diagonal
        c = np.zeros(nx, dtype=np.float64)  # super-diagonal

        for j in range(1, nx - 1):
            a[j] = mu[j - 1] / (2.0 * dx) + 0.5 * D[j - 1] / dx**2
            b[j] = -D[j] / dx**2
            c[j] = -mu[j + 1] / (2.0 * dx) + 0.5 * D[j + 1] / dx**2

        # Reflecting boundary conditions:
        # Zero flux: J = μp - ½ d/dx[Dp] = 0 at boundaries
        # Approximate: p_0 = p_1, p_{nx-1} = p_{nx-2} (Neumann)
        b[0] = b[1]
        c[0] = c[1]
        a[nx - 1] = a[nx - 2]
        b[nx - 1] = b[nx - 2]

        # Time stepping with Crank-Nicolson
        # (I - dt/2 * L) p^{n+1} = (I + dt/2 * L) p^n
        #
        # This gives a tridiagonal system:
        # Left side:  (-dt/2 * a_j) p_{j-1} + (1 - dt/2 * b_j) p_j + (-dt/2 * c_j) p_{j+1}
        # Right side: (dt/2 * a_j) p_{j-1} + (1 + dt/2 * b_j) p_j + (dt/2 * c_j) p_{j+1}

        r = dt / 2.0

        for n in range(1, nt):
            # Build right-hand side
            rhs = np.zeros(nx, dtype=np.float64)
            for j in range(1, nx - 1):
                rhs[j] = (r * a[j]) * p[j - 1] + (1.0 + r * b[j]) * p[j] + (r * c[j]) * p[j + 1]

            # Boundary: reflecting
            rhs[0] = (1.0 + r * b[0]) * p[0] + (r * c[0]) * p[1]
            rhs[nx - 1] = (r * a[nx - 1]) * p[nx - 2] + (1.0 + r * b[nx - 1]) * p[nx - 1]

            # Solve tridiagonal system (Thomas algorithm)
            # System: -r*a_j * p_{j-1} + (1 - r*b_j) * p_j + (-r*c_j) * p_{j+1} = rhs_j
            sub = np.zeros(nx, dtype=np.float64)
            diag = np.zeros(nx, dtype=np.float64)
            sup = np.zeros(nx, dtype=np.float64)

            for j in range(1, nx - 1):
                sub[j] = -r * a[j]
                diag[j] = 1.0 - r * b[j]
                sup[j] = -r * c[j]

            # Boundary entries
            diag[0] = 1.0 - r * b[0]
            sup[0] = -r * c[0]
            sub[nx - 1] = -r * a[nx - 1]
            diag[nx - 1] = 1.0 - r * b[nx - 1]

            p = _thomas_solve(sub, diag, sup, rhs)

            # Enforce non-negativity and renormalize
            p = np.maximum(p, 0.0)
            norm = np.trapz(p, x_grid)
            if norm > 0:
                p /= norm

            density[n] = p

        # Compute steady state from final density
        steady = density[-1].copy()

        return FPESolution(
            density=density,
            x_grid=x_grid,
            t_grid=t_grid,
            steady_state=steady
        )

    @staticmethod
    def steady_state(
        drift_fn: Callable[[float], float],
        diffusion_fn: Callable[[float], float],
        x_range: tuple[float, float],
        nx: int = 200
    ) -> np.ndarray:
        """
        Compute the steady-state density directly.

        For dp/dt = 0 with zero-flux boundaries (detailed balance):
          p_ss(x) = C / D(x) * exp(2 ∫_{x_0}^{x} μ(x') / D(x') dx')

        where C is a normalization constant.

        Args:
            drift_fn: Drift coefficient μ(x).
            diffusion_fn: Diffusion coefficient D(x).
            x_range: Spatial domain (x_min, x_max).
            nx: Number of grid points.

        Returns:
            Steady-state density p_ss(x), shape (nx,).
        """
        x_min, x_max = x_range
        x_grid = np.linspace(x_min, x_max, nx)
        dx = (x_max - x_min) / (nx - 1)

        mu = np.array([drift_fn(x) for x in x_grid])
        D = np.array([diffusion_fn(x) for x in x_grid])
        D = np.maximum(D, 1e-15)

        # Compute the potential: Φ(x) = -2 ∫ μ(x)/D(x) dx
        # p_ss(x) ∝ (1/D(x)) * exp(-Φ(x))
        integrand = 2.0 * mu / D
        phi = np.zeros(nx, dtype=np.float64)
        for i in range(1, nx):
            phi[i] = phi[i - 1] + 0.5 * (integrand[i] + integrand[i - 1]) * dx

        # Subtract max for numerical stability before exp
        phi_shifted = phi - phi.max()
        p_ss = (1.0 / D) * np.exp(phi_shifted)

        # Normalize
        norm = np.trapz(p_ss, x_grid)
        if norm > 0:
            p_ss /= norm

        return p_ss

    @staticmethod
    def mean_first_passage_time(
        drift_fn: Callable[[float], float],
        diffusion_fn: Callable[[float], float],
        x_range: tuple[float, float],
        target: float,
        nx: int = 200
    ) -> float:
        """
        Compute the mean first passage time (MFPT) to reach a target.

        The MFPT T(x) from starting point x_0 = x_range[0] to target
        satisfies the backward Kolmogorov equation:
          μ(x) T'(x) + ½ D(x) T''(x) = -1
        with T(target) = 0 and reflecting BC at x_min.

        Numerical solution via finite differences.

        Args:
            drift_fn: Drift coefficient μ(x).
            diffusion_fn: Diffusion coefficient D(x).
            x_range: Domain (start, beyond_target). Must contain target.
            target: Absorbing boundary position.
            nx: Number of grid points.

        Returns:
            Mean first passage time from x_range[0] to target.
        """
        x_min, x_max = x_range
        x_grid = np.linspace(x_min, x_max, nx)
        dx = (x_max - x_min) / (nx - 1)

        # Find grid index closest to target
        target_idx = np.argmin(np.abs(x_grid - target))
        if target_idx == 0:
            return 0.0

        mu = np.array([drift_fn(x) for x in x_grid])
        D = np.array([diffusion_fn(x) for x in x_grid])
        D = np.maximum(D, 1e-15)

        # Solve on [x_min, target]: μ T' + ½ D T'' = -1
        # Central differences:
        #   T'_j = (T_{j+1} - T_{j-1}) / (2dx)
        #   T''_j = (T_{j+1} - 2T_j + T_{j-1}) / dx²
        #
        # a_j T_{j-1} + b_j T_j + c_j T_{j+1} = -1
        # a_j = -μ_j/(2dx) + ½ D_j/dx²
        # b_j = -D_j/dx²
        # c_j = μ_j/(2dx) + ½ D_j/dx²

        n = target_idx + 1  # grid points from x_min to target (inclusive)

        # Build tridiagonal system
        sub = np.zeros(n, dtype=np.float64)
        diag = np.zeros(n, dtype=np.float64)
        sup = np.zeros(n, dtype=np.float64)
        rhs = np.full(n, -1.0, dtype=np.float64)

        for j in range(1, n - 1):
            sub[j] = -mu[j] / (2.0 * dx) + 0.5 * D[j] / dx**2
            diag[j] = -D[j] / dx**2
            sup[j] = mu[j] / (2.0 * dx) + 0.5 * D[j] / dx**2

        # BC: T(target) = 0 (absorbing)
        diag[n - 1] = 1.0
        sub[n - 1] = 0.0
        rhs[n - 1] = 0.0

        # BC: reflecting at x_min → T'(x_min) = 0 → T_0 = T_1
        # Substitute T_0 = T_1 into equation for j=0
        diag[0] = -D[0] / dx**2 + (-mu[0] / (2.0 * dx) + 0.5 * D[0] / dx**2)
        sup[0] = mu[0] / (2.0 * dx) + 0.5 * D[0] / dx**2

        T = _thomas_solve(sub, diag, sup, rhs)

        return float(T[0])

    @staticmethod
    def ornstein_uhlenbeck_solution(
        theta: float,
        mu: float,
        sigma: float,
        x_range: tuple[float, float],
        t: float,
        x0: float = 0.0,
        nx: int = 200
    ) -> np.ndarray:
        """
        Exact Gaussian solution for the Ornstein-Uhlenbeck process.

        The O-U process: dX = θ(μ - X)dt + σ dW
        has the exact transition density:
          p(x, t | x_0) = N(m(t), s²(t))
        where:
          m(t) = μ + (x_0 - μ)e^{-θt}
          s²(t) = σ²/(2θ) (1 - e^{-2θt})

        Args:
            theta: Mean reversion rate (> 0).
            mu: Long-term mean.
            sigma: Volatility.
            x_range: Spatial domain (x_min, x_max).
            t: Time at which to evaluate density.
            x0: Initial position.
            nx: Number of grid points.

        Returns:
            Density p(x, t | x_0) evaluated on grid, shape (nx,).
        """
        x_min, x_max = x_range
        x_grid = np.linspace(x_min, x_max, nx)

        if theta <= 0:
            raise ValueError("Mean reversion rate theta must be positive")

        mean_t = mu + (x0 - mu) * np.exp(-theta * t)
        var_t = (sigma**2 / (2.0 * theta)) * (1.0 - np.exp(-2.0 * theta * t))
        var_t = max(var_t, 1e-15)

        # Gaussian density
        p = (1.0 / np.sqrt(2.0 * np.pi * var_t)) * np.exp(
            -0.5 * (x_grid - mean_t)**2 / var_t
        )

        return p

    @staticmethod
    def kramers_escape_rate(
        barrier_height: float,
        diffusion: float,
        curvature_well: float,
        curvature_barrier: float
    ) -> float:
        """
        Kramers' escape rate over a potential barrier.

        For a particle in a metastable potential well with thermal noise,
        the escape rate is:

          r = (ω_0 * ω_b) / (2π) * exp(-V_b / D)

        where:
          ω_0 = √(curvature_well)     frequency at the bottom of the well
          ω_b = √(curvature_barrier)   frequency at the top of the barrier
          V_b = barrier_height          barrier height
          D = diffusion                 noise intensity (= k_B T in physics)

        Valid when barrier_height >> diffusion (high-barrier limit).

        Args:
            barrier_height: Height of the potential barrier V_b.
            diffusion: Diffusion coefficient D (noise intensity).
            curvature_well: |V''(x_min)| curvature at the potential minimum.
            curvature_barrier: |V''(x_max)| curvature at the barrier top.

        Returns:
            Kramers escape rate r.
        """
        if diffusion <= 0:
            raise ValueError("Diffusion must be positive")
        if curvature_well <= 0 or curvature_barrier <= 0:
            raise ValueError("Curvatures must be positive")

        omega_0 = np.sqrt(curvature_well)
        omega_b = np.sqrt(curvature_barrier)

        rate = (omega_0 * omega_b / (2.0 * np.pi)) * np.exp(-barrier_height / diffusion)

        return float(rate)

    @staticmethod
    def probability_current(
        density: np.ndarray,
        drift_fn: Callable[[float], float],
        diffusion_fn: Callable[[float], float],
        x_grid: np.ndarray
    ) -> np.ndarray:
        """
        Compute the probability current J(x, t).

        J(x) = μ(x)p(x) - ½ d/dx[D(x)p(x)]

        At steady state with reflecting boundaries, J = 0 (detailed balance).

        Args:
            density: Probability density p(x), shape (nx,).
            drift_fn: Drift coefficient μ(x).
            diffusion_fn: Diffusion coefficient D(x).
            x_grid: Spatial grid, shape (nx,).

        Returns:
            Probability current J(x), shape (nx,).
        """
        nx = len(x_grid)
        dx = x_grid[1] - x_grid[0]

        mu = np.array([drift_fn(x) for x in x_grid])
        D = np.array([diffusion_fn(x) for x in x_grid])

        Dp = D * density

        # J = μ*p - ½ d(Dp)/dx
        J = np.zeros(nx, dtype=np.float64)

        # Central differences for interior
        for j in range(1, nx - 1):
            dDp_dx = (Dp[j + 1] - Dp[j - 1]) / (2.0 * dx)
            J[j] = mu[j] * density[j] - 0.5 * dDp_dx

        # Forward/backward for boundaries
        J[0] = mu[0] * density[0] - 0.5 * (Dp[1] - Dp[0]) / dx
        J[nx - 1] = mu[nx - 1] * density[nx - 1] - 0.5 * (Dp[nx - 1] - Dp[nx - 2]) / dx

        return J

    @staticmethod
    def relaxation_time(
        drift_fn: Callable[[float], float],
        diffusion_fn: Callable[[float], float],
        x_range: tuple[float, float],
        nx: int = 200
    ) -> float:
        """
        Estimate the relaxation time to steady state.

        The relaxation time is τ = 1/|λ_2| where λ_2 is the second
        eigenvalue of the FPE operator (largest non-zero in magnitude).
        Computed via discretized operator eigenvalue analysis.

        Args:
            drift_fn: Drift coefficient μ(x).
            diffusion_fn: Diffusion coefficient D(x).
            x_range: Spatial domain.
            nx: Number of grid points.

        Returns:
            Relaxation time τ.
        """
        x_min, x_max = x_range
        x_grid = np.linspace(x_min, x_max, nx)
        dx = (x_max - x_min) / (nx - 1)

        mu = np.array([drift_fn(x) for x in x_grid])
        D = np.array([diffusion_fn(x) for x in x_grid])
        D = np.maximum(D, 1e-15)

        # Build FPE operator matrix L (nx x nx)
        L = np.zeros((nx, nx), dtype=np.float64)

        for j in range(1, nx - 1):
            a_j = mu[j - 1] / (2.0 * dx) + 0.5 * D[j - 1] / dx**2
            b_j = -D[j] / dx**2
            c_j = -mu[j + 1] / (2.0 * dx) + 0.5 * D[j + 1] / dx**2
            L[j, j - 1] = a_j
            L[j, j] = b_j
            L[j, j + 1] = c_j

        # Reflecting BC
        L[0, 0] = L[1, 1]
        L[0, 1] = L[1, 2] if nx > 2 else 0
        L[nx - 1, nx - 2] = L[nx - 2, nx - 3] if nx > 2 else 0
        L[nx - 1, nx - 1] = L[nx - 2, nx - 2]

        eigenvalues = np.linalg.eigvals(L)
        real_parts = np.real(eigenvalues)

        # Sort by real part (descending). First should be ~0 (steady state).
        sorted_eigs = np.sort(real_parts)[::-1]

        # λ_2 is the second largest (most negative non-zero)
        # Find the first eigenvalue that is clearly negative
        for eig in sorted_eigs:
            if eig < -1e-10:
                return float(-1.0 / eig)

        # If no negative eigenvalue found, system is degenerate
        logger.warning("No negative eigenvalue found — system may be degenerate")
        return float('inf')


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _thomas_solve(
    sub: np.ndarray,
    diag: np.ndarray,
    sup: np.ndarray,
    rhs: np.ndarray
) -> np.ndarray:
    """
    Thomas algorithm for tridiagonal systems Ax = d.

    Solves in O(n) time. The tridiagonal matrix has:
      sub[i] on the sub-diagonal (sub[0] unused)
      diag[i] on the main diagonal
      sup[i] on the super-diagonal (sup[n-1] unused)

    Args:
        sub: Sub-diagonal, shape (n,).
        diag: Main diagonal, shape (n,).
        sup: Super-diagonal, shape (n,).
        rhs: Right-hand side, shape (n,).

    Returns:
        Solution vector x, shape (n,).
    """
    n = len(diag)
    # Copy to avoid modifying inputs
    c = sup.copy()
    d = rhs.copy()
    b = diag.copy()
    a = sub.copy()

    # Forward sweep
    for i in range(1, n):
        if abs(b[i - 1]) < 1e-300:
            b[i - 1] = 1e-300
        m = a[i] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]

    # Back substitution
    x = np.zeros(n, dtype=np.float64)
    if abs(b[n - 1]) < 1e-300:
        b[n - 1] = 1e-300
    x[n - 1] = d[n - 1] / b[n - 1]

    for i in range(n - 2, -1, -1):
        if abs(b[i]) < 1e-300:
            b[i] = 1e-300
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'FPESolution',
    'FokkerPlanckSolver',
]
