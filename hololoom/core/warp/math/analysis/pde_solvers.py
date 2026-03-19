"""
Partial Differential Equation Solvers
======================================

Scope slots: DE4.3 (wave equation), DE4.4 (Laplace equation),
DE4.5 (finite element method).

Pure numpy implementations:
- Wave equation solver (1D leapfrog / Verlet scheme)
- Laplace equation solver (2D iterative relaxation / Gauss-Seidel)
- Finite element method (1D linear elements for -d/dx[a(x) du/dx] = f(x))

Author: Claude Code
Date: March 2026
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# WAVE EQUATION SOLVER — 1D leapfrog (DE4.3)
# ============================================================================

@dataclass
class WaveResult:
    """Result of wave equation solve.

    Attributes:
        x: Spatial grid of shape (nx,).
        t: Time grid of shape (nt,).
        u: Solution u(x, t) of shape (nx, nt).
    """
    x: np.ndarray
    t: np.ndarray
    u: np.ndarray


class WaveEquationSolver:
    """Solve the 1D wave equation u_tt = c² u_xx via leapfrog.

    The leapfrog (Verlet) scheme is second-order in both space and time
    and is time-reversible:

        u^{n+1}_j = 2u^n_j - u^{n-1}_j + r²(u^n_{j+1} - 2u^n_j + u^n_{j-1})

    where r = c dt/dx is the Courant number. Stable when r ≤ 1 (CFL condition).

    Example:
        >>> solver = WaveEquationSolver()
        >>> # Plucked string: triangle initial displacement
        >>> x = np.linspace(0, 1, 101)
        >>> u0 = np.where(x < 0.5, 2*x, 2*(1-x))
        >>> result = solver.solve_1d(c=1.0, initial_u=u0, initial_v=np.zeros_like(u0),
        ...                          dx=0.01, dt=0.005, n_steps=200)
    """

    def solve_1d(
        self,
        c: float,
        initial_u: np.ndarray,
        initial_v: np.ndarray,
        dx: float,
        dt: float,
        n_steps: int,
        boundary: str = 'dirichlet',
        save_every: int = 1,
    ) -> WaveResult:
        """Solve 1D wave equation u_tt = c² u_xx.

        Args:
            c: Wave speed.
            initial_u: Initial displacement u(x, 0) of shape (nx,).
            initial_v: Initial velocity u_t(x, 0) of shape (nx,).
            dx: Spatial step size.
            dt: Time step size.
            n_steps: Number of time steps.
            boundary: 'dirichlet' (fixed ends) or 'neumann' (free ends).
            save_every: Save solution every N steps (1 = all).

        Returns:
            WaveResult with spatial grid, time grid, and solution.

        Raises:
            ValueError: If CFL condition is violated (r > 1).
        """
        nx = len(initial_u)
        r = c * dt / dx  # Courant number
        r2 = r * r

        if r > 1.0:
            raise ValueError(
                f"CFL condition violated: Courant number r = {r:.4f} > 1. "
                f"Reduce dt or increase dx."
            )

        x = np.arange(nx) * dx

        # Initialize: u^0 = initial_u, u^1 from Taylor expansion
        u_prev = initial_u.copy()
        u_curr = initial_u + initial_v * dt + 0.5 * r2 * (
            np.roll(initial_u, -1) - 2 * initial_u + np.roll(initial_u, 1)
        )

        # Apply boundary conditions to u^1
        if boundary == 'dirichlet':
            u_curr[0] = 0.0
            u_curr[-1] = 0.0

        save_indices = list(range(0, n_steps + 1, save_every))
        n_saved = len(save_indices)
        u_saved = np.zeros((nx, n_saved))
        t_saved = np.zeros(n_saved)

        save_idx = 0
        if 0 in save_indices:
            u_saved[:, 0] = u_prev
            t_saved[0] = 0.0
            save_idx = 1

        for step in range(1, n_steps + 1):
            # Leapfrog update
            u_next = np.zeros(nx)
            u_next[1:-1] = (
                2 * u_curr[1:-1] - u_prev[1:-1]
                + r2 * (u_curr[2:] - 2 * u_curr[1:-1] + u_curr[:-2])
            )

            # Boundary conditions
            if boundary == 'dirichlet':
                u_next[0] = 0.0
                u_next[-1] = 0.0
            elif boundary == 'neumann':
                u_next[0] = u_next[1]
                u_next[-1] = u_next[-2]

            u_prev = u_curr
            u_curr = u_next

            if step in save_indices and save_idx < n_saved:
                u_saved[:, save_idx] = u_curr
                t_saved[save_idx] = step * dt
                save_idx += 1

        return WaveResult(x=x, t=t_saved[:save_idx], u=u_saved[:, :save_idx])


# ============================================================================
# LAPLACE EQUATION SOLVER — 2D relaxation (DE4.4)
# ============================================================================

@dataclass
class LaplaceResult:
    """Result of Laplace equation solve.

    Attributes:
        u: Solution u(x, y) of shape (ny, nx).
        iterations: Number of iterations to convergence.
        residual: Final residual norm.
    """
    u: np.ndarray
    iterations: int
    residual: float


class LaplaceEquationSolver:
    """Solve 2D Laplace equation ∇²u = 0 via iterative relaxation.

    Uses Gauss-Seidel iteration with successive over-relaxation (SOR).
    The optimal relaxation parameter for an n×n grid is approximately:
        ω_opt = 2 / (1 + sin(π/n))

    Example:
        >>> solver = LaplaceEquationSolver()
        >>> # Unit square, top boundary = sin(πx), others = 0
        >>> boundary = {'top': lambda x: np.sin(np.pi * x),
        ...             'bottom': 0.0, 'left': 0.0, 'right': 0.0}
        >>> result = solver.solve_2d(boundary, nx=50, ny=50, tol=1e-6)
    """

    def solve_2d(
        self,
        boundary: dict,
        nx: int = 50,
        ny: int = 50,
        tol: float = 1e-6,
        max_iter: int = 10000,
        omega: float | None = None,
    ) -> LaplaceResult:
        """Solve ∇²u = 0 on [0,1]×[0,1] with given boundary conditions.

        Args:
            boundary: Dict with keys 'top', 'bottom', 'left', 'right'.
                Values can be float (constant), np.ndarray (values), or
                callable (function of position along that edge).
            nx: Number of interior grid points in x.
            ny: Number of interior grid points in y.
            tol: Convergence tolerance on max residual.
            max_iter: Maximum iterations.
            omega: SOR relaxation parameter (auto-computed if None).

        Returns:
            LaplaceResult with solution, iteration count, and residual.
        """
        # Total grid including boundary
        Nx = nx + 2
        Ny = ny + 2
        u = np.zeros((Ny, Nx))

        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)

        # Apply boundary conditions
        u[-1, :] = self._eval_bc(boundary.get('top', 0.0), x)       # top row
        u[0, :] = self._eval_bc(boundary.get('bottom', 0.0), x)     # bottom row
        u[:, 0] = self._eval_bc(boundary.get('left', 0.0), y)       # left col
        u[:, -1] = self._eval_bc(boundary.get('right', 0.0), y)     # right col

        # Optimal SOR parameter
        if omega is None:
            omega = 2.0 / (1.0 + np.sin(np.pi / max(nx, ny)))

        # Gauss-Seidel with SOR
        for iteration in range(1, max_iter + 1):
            max_residual = 0.0

            for j in range(1, Ny - 1):
                for i in range(1, Nx - 1):
                    u_gs = 0.25 * (u[j+1, i] + u[j-1, i] + u[j, i+1] + u[j, i-1])
                    residual = abs(u_gs - u[j, i])
                    u[j, i] = u[j, i] + omega * (u_gs - u[j, i])
                    max_residual = max(max_residual, residual)

            if max_residual < tol:
                return LaplaceResult(u=u, iterations=iteration, residual=max_residual)

        logger.warning(f"Laplace solver did not converge in {max_iter} iterations "
                       f"(residual: {max_residual:.2e})")
        return LaplaceResult(u=u, iterations=max_iter, residual=max_residual)

    def solve_poisson_2d(
        self,
        source: np.ndarray,
        boundary: dict,
        dx: float = 1.0,
        dy: float = 1.0,
        tol: float = 1e-6,
        max_iter: int = 10000,
    ) -> LaplaceResult:
        """Solve 2D Poisson equation ∇²u = f on a grid.

        Extends the Laplace solver to handle a source term.

        Args:
            source: Source term f(x, y) on the interior grid.
            boundary: Boundary conditions (same format as solve_2d).
            dx, dy: Grid spacings.
            tol: Convergence tolerance.
            max_iter: Maximum iterations.

        Returns:
            LaplaceResult.
        """
        ny_int, nx_int = source.shape
        Nx = nx_int + 2
        Ny = ny_int + 2
        u = np.zeros((Ny, Nx))

        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)

        u[-1, :] = self._eval_bc(boundary.get('top', 0.0), x)
        u[0, :] = self._eval_bc(boundary.get('bottom', 0.0), x)
        u[:, 0] = self._eval_bc(boundary.get('left', 0.0), y)
        u[:, -1] = self._eval_bc(boundary.get('right', 0.0), y)

        coeff = 2.0 * (1.0 / dx**2 + 1.0 / dy**2)

        for iteration in range(1, max_iter + 1):
            max_residual = 0.0

            for j in range(1, Ny - 1):
                for i in range(1, Nx - 1):
                    u_new = (
                        (u[j, i-1] + u[j, i+1]) / dx**2
                        + (u[j-1, i] + u[j+1, i]) / dy**2
                        - source[j-1, i-1]
                    ) / coeff
                    residual = abs(u_new - u[j, i])
                    u[j, i] = u_new
                    max_residual = max(max_residual, residual)

            if max_residual < tol:
                return LaplaceResult(u=u, iterations=iteration, residual=max_residual)

        return LaplaceResult(u=u, iterations=max_iter, residual=max_residual)

    @staticmethod
    def _eval_bc(bc, positions: np.ndarray) -> np.ndarray:
        """Evaluate boundary condition."""
        if callable(bc):
            return np.array([bc(p) for p in positions])
        elif isinstance(bc, np.ndarray):
            return bc
        else:
            return np.full_like(positions, float(bc))


# ============================================================================
# FINITE ELEMENT METHOD — 1D linear elements (DE4.5)
# ============================================================================

@dataclass
class FEMResult:
    """Result of finite element solve.

    Attributes:
        nodes: Node positions of shape (n_nodes,).
        solution: Solution values at nodes of shape (n_nodes,).
        stiffness_matrix: Global stiffness matrix.
        load_vector: Global load vector.
    """
    nodes: np.ndarray
    solution: np.ndarray
    stiffness_matrix: np.ndarray
    load_vector: np.ndarray


class FiniteElementMethod:
    """1D Finite Element Method with linear (hat) basis functions.

    Solves the BVP:  -d/dx[a(x) du/dx] = f(x)  on [x_L, x_R]
    with boundary conditions u(x_L) = u_L, u(x_R) = u_R.

    Uses piecewise linear (P1) elements and Gauss quadrature for
    element stiffness and load assembly.

    The weak form: find u ∈ V_h such that
        ∫ a(x) u'(x) v'(x) dx = ∫ f(x) v(x) dx  ∀ v ∈ V_h⁰

    Example:
        >>> fem = FiniteElementMethod()
        >>> # -u'' = 1 on [0,1], u(0) = u(1) = 0  →  u = x(1-x)/2
        >>> result = fem.solve_1d(
        ...     a_fn=lambda x: 1.0,
        ...     f_fn=lambda x: 1.0,
        ...     boundary=(0.0, 0.0),
        ...     n_elements=20
        ... )
        >>> abs(result.solution[10] - 0.125) < 0.01  # u(0.5) ≈ 0.125
        True
    """

    def solve_1d(
        self,
        a_fn: Callable[[float], float],
        f_fn: Callable[[float], float],
        boundary: tuple[float, float],
        n_elements: int = 20,
        domain: tuple[float, float] = (0.0, 1.0),
        n_gauss: int = 2,
    ) -> FEMResult:
        """Solve 1D BVP with linear finite elements.

        Args:
            a_fn: Coefficient function a(x).
            f_fn: Source function f(x).
            boundary: (u_left, u_right) Dirichlet boundary values.
            n_elements: Number of elements.
            domain: (x_left, x_right) spatial domain.
            n_gauss: Number of Gauss quadrature points per element.

        Returns:
            FEMResult with nodes, solution, stiffness matrix, and load vector.
        """
        x_L, x_R = domain
        u_L, u_R = boundary
        n_nodes = n_elements + 1

        # Generate uniform mesh
        nodes = np.linspace(x_L, x_R, n_nodes)

        # Gauss quadrature points and weights on [-1, 1]
        gauss_pts, gauss_wts = np.polynomial.legendre.leggauss(n_gauss)

        # Global stiffness matrix and load vector
        K = np.zeros((n_nodes, n_nodes))
        F = np.zeros(n_nodes)

        for e in range(n_elements):
            # Element nodes
            x1, x2 = nodes[e], nodes[e + 1]
            h = x2 - x1  # Element length

            # Local stiffness and load
            K_local = np.zeros((2, 2))
            F_local = np.zeros(2)

            for gp, gw in zip(gauss_pts, gauss_wts):
                # Map Gauss point to physical element: x = (x1+x2)/2 + gp*(x2-x1)/2
                x_phys = 0.5 * (x1 + x2) + 0.5 * gp * h

                # Shape functions on reference element [-1, 1]
                N1 = 0.5 * (1.0 - gp)
                N2 = 0.5 * (1.0 + gp)

                # Shape function derivatives (w.r.t. physical x)
                dN1_dx = -1.0 / h
                dN2_dx = 1.0 / h

                # Jacobian
                J = 0.5 * h

                # Coefficient and source at quadrature point
                a_val = a_fn(x_phys)
                f_val = f_fn(x_phys)

                # Stiffness contribution
                K_local[0, 0] += a_val * dN1_dx * dN1_dx * J * gw
                K_local[0, 1] += a_val * dN1_dx * dN2_dx * J * gw
                K_local[1, 0] += a_val * dN2_dx * dN1_dx * J * gw
                K_local[1, 1] += a_val * dN2_dx * dN2_dx * J * gw

                # Load contribution
                F_local[0] += f_val * N1 * J * gw
                F_local[1] += f_val * N2 * J * gw

            # Assemble into global system
            dof = [e, e + 1]
            for i in range(2):
                F[dof[i]] += F_local[i]
                for j in range(2):
                    K[dof[i], dof[j]] += K_local[i, j]

        # Apply Dirichlet boundary conditions
        K_full = K.copy()
        F_full = F.copy()

        # Left BC
        K[0, :] = 0.0
        K[0, 0] = 1.0
        F[0] = u_L

        # Right BC
        K[-1, :] = 0.0
        K[-1, -1] = 1.0
        F[-1] = u_R

        # Modify RHS for non-homogeneous BCs
        for i in range(1, n_nodes - 1):
            F[i] -= K_full[i, 0] * u_L + K_full[i, -1] * u_R

        # Solve
        solution = np.linalg.solve(K, F)

        return FEMResult(
            nodes=nodes,
            solution=solution,
            stiffness_matrix=K_full,
            load_vector=F_full,
        )

    @staticmethod
    def l2_error(
        nodes: np.ndarray,
        numerical: np.ndarray,
        exact: Callable[[np.ndarray], np.ndarray],
    ) -> float:
        """Compute L2 error between numerical and exact solution.

        Uses trapezoidal rule for integration.

        Args:
            nodes: Node positions.
            numerical: Numerical solution at nodes.
            exact: Exact solution function.

        Returns:
            L2 norm of the error.
        """
        exact_vals = exact(nodes)
        error = numerical - exact_vals
        dx = np.diff(nodes)
        # Trapezoidal rule
        l2_sq = np.sum(0.5 * dx * (error[:-1]**2 + error[1:]**2))
        return np.sqrt(l2_sq)
