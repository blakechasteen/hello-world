"""
PDE-Based Semantic Flow
=======================
Partial differential equations for modeling semantic information flow.

WHY PDEs FOR SEMANTICS?
- Discrete graph → Continuous manifold (as n → ∞)
- Natural models for diffusion, wave propagation, reaction
- Rich theory: existence, uniqueness, stability
- Connects to physics (Hamilton-Jacobi, Navier-Stokes)

BREAKTHROUGH APPLICATIONS:
1. **Heat Equation**: Information diffusion through knowledge graph
2. **Wave Equation**: Semantic oscillations and resonance
3. **Reaction-Diffusion**: Competitive activation (e.g., disambiguation)
4. **Hamilton-Jacobi**: Optimal paths in semantic space

Mathematical Foundation:
-----------------------
Classical PDEs on graphs → Continuum limit:

1. Heat Equation:
   ∂u/∂t = Δu
   where Δ = Laplacian, u = activation

2. Wave Equation:
   ∂²u/∂t² = c² Δu
   Models oscillatory phenomena

3. Reaction-Diffusion:
   ∂u/∂t = D Δu + f(u)
   Combines diffusion + nonlinear dynamics

4. Hamilton-Jacobi:
   ∂u/∂t + H(∇u) = 0
   Optimal control and geodesics

Author: HoloLoom PDE Team
Date: 2025-11-03
"""

from typing import Callable, Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
import warnings


# ============================================================================
# PDE Types
# ============================================================================

class PDEType(Enum):
    """Types of PDEs for semantic flow."""

    HEAT = "heat"                    # Parabolic: ∂u/∂t = Δu
    WAVE = "wave"                    # Hyperbolic: ∂²u/∂t² = c² Δu
    REACTION_DIFFUSION = "reaction_diffusion"  # Parabolic + nonlinear
    HAMILTON_JACOBI = "hamilton_jacobi"        # First-order nonlinear


@dataclass
class PDEConfig:
    """Lightweight configuration container used by integration tests."""

    dt: float = 0.01
    dx: float = 0.1
    n_steps: int = 100
    diffusion_coefficient: float = 0.1
    wave_speed: float = 1.0
    velocity: float = 1.0


# ============================================================================
# Heat Equation (Diffusion)
# ============================================================================

@dataclass
class HeatEquationSolver:
    """1D heat equation solver with finite-difference compatibility helpers."""

    laplacian: Optional[np.ndarray] = None
    dt: float = 0.01
    implicit: bool = True
    domain_size: float = 1.0
    n_points: int = 101
    diffusion_coefficient: float = 0.1
    boundary_condition: str = "dirichlet"

    def __post_init__(self):
        if self.laplacian is not None:
            self.n_points = self.laplacian.shape[0]
        self.dx = self.domain_size / max(self.n_points - 1, 1)

        if self.laplacian is None:
            self.laplacian = self._build_1d_laplacian(
                self.n_points,
                self.dx,
                self.boundary_condition
            )

        self.n = self.n_points
        self.alpha = self.diffusion_coefficient * self.dt / (self.dx ** 2 if self.dx > 0 else 1.0)

        if self.implicit:
            self.A = np.eye(self.n) - self.diffusion_coefficient * self.dt * self.laplacian
            try:
                self.A_inv = np.linalg.inv(self.A)
            except np.linalg.LinAlgError:
                warnings.warn("Implicit heat solver matrix singular; switching to explicit update")
                self.implicit = False

        if not self.implicit and self.alpha > 0.5:
            warnings.warn(
                "Heat equation CFL condition violated (dt too large). Results may be unstable.",
                RuntimeWarning,
            )

    @staticmethod
    def _build_1d_laplacian(n: int, dx: float, bc: str) -> np.ndarray:
        if n < 2:
            return np.zeros((n, n))

        scale = 1.0 / (dx ** 2 if dx > 0 else 1.0)
        L = np.zeros((n, n))
        np.fill_diagonal(L, -2.0 * scale)
        indices = np.arange(n - 1)
        L[indices, indices + 1] = scale
        L[indices + 1, indices] = scale

        if bc == "periodic":
            L[0, -1] = scale
            L[-1, 0] = scale
        elif bc == "neumann":
            L[0, 0] = -1.0 * scale
            L[-1, -1] = -1.0 * scale
            L[0, 1] = scale
            L[-1, -2] = scale

        return L

    def _apply_boundary(self, u: np.ndarray) -> np.ndarray:
        if len(u) == 0:
            return u

        if self.boundary_condition == "dirichlet":
            u[0] = 0.0
            u[-1] = 0.0
        elif self.boundary_condition == "neumann":
            u[0] = u[1]
            u[-1] = u[-2]
        elif self.boundary_condition == "periodic":
            u[0] = u[-2]
            u[-1] = u[1]
        return u

    def step(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float).reshape(-1)

        if self.implicit:
            next_u = self.A_inv @ u
        else:
            diffusion = self.laplacian @ u
            next_u = u + self.diffusion_coefficient * self.dt * diffusion

        return self._apply_boundary(next_u.copy())

    def solve(self, u0: np.ndarray, t_final: float, n_snapshots: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        n_steps = int(max(t_final / self.dt, 1))
        snapshot_interval = max(1, n_steps // max(n_snapshots, 1))

        times = []
        solutions = []

        u = np.asarray(u0, dtype=float).reshape(-1)
        t = 0.0

        for step in range(n_steps):
            if step % snapshot_interval == 0:
                times.append(t)
                solutions.append(u.copy())

            u = self.step(u)
            t += self.dt

        times.append(t_final)
        solutions.append(u.copy())

        return np.array(times), np.array(solutions)


# ============================================================================
# Wave Equation
# ============================================================================

@dataclass
class WaveEquationSolver:
    """1D wave equation solver supporting legacy graph-based API."""

    laplacian: Optional[np.ndarray] = None
    wave_speed: float = 1.0
    dt: float = 0.01
    domain_size: float = 1.0
    n_points: int = 101
    boundary_condition: str = "dirichlet"

    def __post_init__(self):
        if self.laplacian is not None:
            self.n_points = self.laplacian.shape[0]

        self.dx = self.domain_size / max(self.n_points - 1, 1)

        if self.laplacian is None:
            self.laplacian = HeatEquationSolver._build_1d_laplacian(
                self.n_points,
                self.dx,
                self.boundary_condition
            )

        cfl = self.wave_speed * self.dt / (self.dx if self.dx > 0 else 1.0)
        if cfl > 1.0:
            warnings.warn(
                f"Wave equation Courant condition violated (c*dt/dx={cfl:.2f}). Expect oscillations.",
                RuntimeWarning,
            )

    def _apply_boundary(self, u: np.ndarray) -> np.ndarray:
        if len(u) == 0:
            return u
        if self.boundary_condition == "dirichlet":
            u[0] = 0.0
            u[-1] = 0.0
        elif self.boundary_condition == "neumann":
            u[0] = u[1]
            u[-1] = u[-2]
        elif self.boundary_condition == "periodic":
            u[0] = u[-2]
            u[-1] = u[1]
        return u

    def step(self, u: np.ndarray, ut: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u = np.asarray(u, dtype=float).reshape(-1)
        ut = np.asarray(ut, dtype=float).reshape(-1)

        laplacian_u = self.laplacian @ u
        u_new = u + self.dt * ut + 0.5 * (self.wave_speed ** 2) * (self.dt ** 2) * laplacian_u
        ut_new = ut + (self.wave_speed ** 2) * self.dt * laplacian_u

        return self._apply_boundary(u_new.copy()), self._apply_boundary(ut_new.copy())

    def solve(self, u0: np.ndarray, v0: np.ndarray, t_final: float, n_snapshots: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        n_steps = int(max(t_final / self.dt, 1))
        snapshot_interval = max(1, n_steps // max(n_snapshots, 1))

        times = [0.0]
        solutions = [np.asarray(u0, dtype=float).reshape(-1).copy()]

        u = np.asarray(u0, dtype=float).reshape(-1)
        ut = np.asarray(v0, dtype=float).reshape(-1)

        for step in range(1, n_steps + 1):
            u, ut = self.step(u, ut)
            if step % snapshot_interval == 0:
                times.append(step * self.dt)
                solutions.append(u.copy())

        if times[-1] < t_final:
            times.append(t_final)
            solutions.append(u.copy())

        return np.array(times), np.array(solutions)


# ============================================================================
# Reaction-Diffusion
# ============================================================================

@dataclass
class ReactionDiffusionSolver:
    """1D two-species reaction-diffusion (Gray-Scott inspired) solver."""

    laplacian: Optional[np.ndarray] = None
    domain_size: float = 1.0
    n_points: int = 51
    diffusion_a: float = 0.1
    diffusion_b: float = 0.05
    reaction_rate_a: float = 0.04  # feed rate
    reaction_rate_b: float = 0.06  # kill rate
    dt: float = 0.01

    def __post_init__(self):
        if self.laplacian is not None:
            self.n_points = self.laplacian.shape[0]
        self.dx = self.domain_size / max(self.n_points - 1, 1)

        if self.laplacian is None:
            self.laplacian = HeatEquationSolver._build_1d_laplacian(
                self.n_points,
                self.dx,
                "neumann"
            )

    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        return self.laplacian @ field

    def step(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u = np.asarray(u, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)

        Lu = self._laplacian(u)
        Lv = self._laplacian(v)

        uvv = u * (v ** 2)

        du = self.diffusion_a * Lu - uvv + self.reaction_rate_a * (1.0 - u)
        dv = self.diffusion_b * Lv + uvv - (self.reaction_rate_a + self.reaction_rate_b) * v

        u_next = np.clip(u + self.dt * du, 0.0, 1.5)
        v_next = np.clip(v + self.dt * dv, 0.0, 1.5)

        # Neumann boundary (zero-gradient)
        u_next[0] = u_next[1]
        u_next[-1] = u_next[-2]
        v_next[0] = v_next[1]
        v_next[-1] = v_next[-2]

        return u_next, v_next

    def solve(self, u0: np.ndarray, v0: np.ndarray, t_final: float, n_snapshots: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        n_steps = int(max(t_final / self.dt, 1))
        snapshot_interval = max(1, n_steps // max(n_snapshots, 1))

        times = [0.0]
        solutions_u = [np.asarray(u0, dtype=float).reshape(-1).copy()]
        solutions_v = [np.asarray(v0, dtype=float).reshape(-1).copy()]

        u = np.asarray(u0, dtype=float).reshape(-1)
        v = np.asarray(v0, dtype=float).reshape(-1)

        for step in range(1, n_steps + 1):
            u, v = self.step(u, v)
            if step % snapshot_interval == 0:
                times.append(step * self.dt)
                solutions_u.append(u.copy())
                solutions_v.append(v.copy())

        if times[-1] < t_final:
            times.append(t_final)
            solutions_u.append(u.copy())
            solutions_v.append(v.copy())

        return np.array(times), np.array(solutions_u), np.array(solutions_v)


@dataclass
class AdvectionDiffusionSolver:
    """1D advection-diffusion solver used by integration tests."""

    domain_size: float = 1.0
    n_points: int = 101
    velocity: float = 1.0
    diffusion_coefficient: float = 0.1
    dt: float = 0.01
    boundary_condition: str = "periodic"

    def __post_init__(self):
        self.dx = self.domain_size / max(self.n_points - 1, 1)

    def _apply_boundary(self, u: np.ndarray) -> np.ndarray:
        if len(u) == 0:
            return u
        if self.boundary_condition == "dirichlet":
            u[0] = 0.0
            u[-1] = 0.0
        elif self.boundary_condition == "neumann":
            u[0] = u[1]
            u[-1] = u[-2]
        elif self.boundary_condition == "periodic":
            u[0] = u[-2]
            u[-1] = u[1]
        return u

    def step(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float).reshape(-1)

        spatial_step = self.dx if self.dx > 0 else 1.0
        advection = -self.velocity * (u - np.roll(u, 1)) / spatial_step
        diffusion = self.diffusion_coefficient * (
            np.roll(u, -1) - 2 * u + np.roll(u, 1)
        ) / (spatial_step ** 2)

        u_next = u + self.dt * (advection + diffusion)
        return self._apply_boundary(u_next)


# ============================================================================
# Hamilton-Jacobi Equation
# ============================================================================

@dataclass
class HamiltonJacobiSolver:
    """
    Solve Hamilton-Jacobi equation: ∂u/∂t + H(∇u) = 0

    Hamiltonian: H(p) = (1/2) ||p||²

    Physical interpretation:
    - u(x, t): Value function (cost-to-go)
    - ∇u: Optimal gradient direction
    - Characteristics: Optimal paths

    Semantic interpretation:
    - u: Semantic distance from target
    - ∇u: Direction of semantic gradient ascent
    - Optimal policy: Follow characteristics

    Discretization (upwind scheme):
    Stable first-order upwind for advection.
    """

    adjacency: np.ndarray            # Adjacency matrix (for gradient)
    hamiltonian: Callable[[np.ndarray], float]
    dt: float = 0.01

    def __post_init__(self):
        self.n = self.adjacency.shape[0]

    def compute_gradient(self, u: np.ndarray) -> np.ndarray:
        """
        Approximate gradient ∇u on graph.

        ∇u[i] ≈ Σ_j (u[j] - u[i]) / d[i,j]
        """
        grad = np.zeros(self.n)

        for i in range(self.n):
            neighbors = np.where(self.adjacency[i] > 0)[0]
            if len(neighbors) > 0:
                grad[i] = np.mean([u[j] - u[i] for j in neighbors])


# ---------------------------------------------------------------------------
# Backwards-compatible helpers and simple PDE config
# ---------------------------------------------------------------------------


@dataclass
class PDEConfig:
    """Simple PDE configuration used by tests."""

    dt: float = 0.01
    dx: float = 0.01
    n_steps: int = 100
    diffusion_coefficient: float = 1.0
    wave_speed: float = 1.0


@dataclass
class AdvectionDiffusionSolver:
    """
    Simple 1D advection-diffusion solver (upwind + central diffusion).

    Implemented for compatibility with integration tests. This solver uses
    periodic boundary conditions for simplicity and stability in tests.
    """

    domain_size: float
    n_points: int
    velocity: float = 1.0
    diffusion_coefficient: float = 0.1
    dt: float = 0.01

    def __post_init__(self):
        self.dx = self.domain_size / (self.n_points - 1)

    def step(self, u: np.ndarray) -> np.ndarray:
        """Advance one time step and return new field (periodic boundaries)."""
        u = np.asarray(u, dtype=float)
        n = self.n_points

        # Pre-allocate
        u_new = np.zeros_like(u)

        # Courant number
        c = self.velocity * self.dt / max(self.dx, 1e-12)

        # Second derivative (central) for diffusion
        lap = np.zeros_like(u)
        for i in range(n):
            ip = (i + 1) % n
            im = (i - 1) % n
            lap[i] = (u[ip] - 2 * u[i] + u[im]) / (self.dx ** 2)

        # Upwind scheme for advection + central diffusion
        if self.velocity >= 0:
            for i in range(n):
                im = (i - 1) % n
                adv = -self.velocity * (u[i] - u[im]) / self.dx
                diff = self.diffusion_coefficient * lap[i]
                u_new[i] = u[i] + self.dt * (adv + diff)
        else:
            for i in range(n):
                ip = (i + 1) % n
                adv = -self.velocity * (u[ip] - u[i]) / self.dx
                diff = self.diffusion_coefficient * lap[i]
                u_new[i] = u[i] + self.dt * (adv + diff)

        return u_new

        return grad

    def step(self, u: np.ndarray) -> np.ndarray:
        """
        Semi-Lagrangian step for Hamilton-Jacobi.

        u^{n+1}[i] = min_j {u^n[j] + dt × H((u[j] - u[i]) / dt)}
        """
        u_next = np.zeros(self.n)

        for i in range(self.n):
            neighbors = list(np.where(self.adjacency[i] > 0)[0])
            neighbors.append(i)  # Include self

            candidates = []
            for j in neighbors:
                p = (u[j] - u[i]) / self.dt  # Approximate gradient
                candidates.append(u[j] + self.dt * self.hamiltonian(p))

            u_next[i] = min(candidates)

        return u_next

    def solve(
        self,
        u0: np.ndarray,
        t_final: float,
        n_snapshots: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve Hamilton-Jacobi equation."""
        n_steps = int(t_final / self.dt)
        snapshot_interval = max(1, n_steps // n_snapshots)

        times = []
        solutions = []

        u = u0.copy()
        times.append(0.0)
        solutions.append(u.copy())

        for step in range(1, n_steps):
            u = self.step(u)

            if step % snapshot_interval == 0:
                times.append(step * self.dt)
                solutions.append(u.copy())

        times.append(t_final)
        solutions.append(u)

        return np.array(times), np.array(solutions)


# ============================================================================
# Common Reaction Functions
# ============================================================================

def logistic_reaction(r: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Logistic growth: f(u) = r u (1 - u)"""
    return lambda u: r * u * (1 - u)


def competitive_reaction(
    r: float = 1.0,
    theta: float = 0.5
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Competitive reaction with threshold.

    f(u) = r u (1 - u) if u > theta else -r u

    Amplifies strong activations, suppresses weak ones.
    """
    def reaction(u):
        mask = u > theta
        return np.where(mask, r * u * (1 - u), -r * u)
    return reaction


def cubic_reaction(a: float = 1.0, b: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Cubic reaction: f(u) = a u - b u³"""
    return lambda u: a * u - b * u**3


# ============================================================================
# Factory Functions
# ============================================================================

def create_heat_solver(laplacian: np.ndarray, **kwargs) -> HeatEquationSolver:
    return HeatEquationSolver(laplacian=laplacian, **kwargs)


def create_wave_solver(laplacian: np.ndarray, **kwargs) -> WaveEquationSolver:
    return WaveEquationSolver(laplacian=laplacian, **kwargs)


def create_reaction_diffusion_solver(
    laplacian: np.ndarray,
    **kwargs
) -> ReactionDiffusionSolver:
    return ReactionDiffusionSolver(
        laplacian=laplacian,
        diffusion_a=kwargs.get('diffusion_a', kwargs.get('diffusion_coef', 0.1)),
        diffusion_b=kwargs.get('diffusion_b', kwargs.get('diffusion_coef_b', 0.05)),
        reaction_rate_a=kwargs.get('reaction_rate_a', 0.04),
        reaction_rate_b=kwargs.get('reaction_rate_b', 0.06),
        dt=kwargs.get('dt', 0.01),
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'PDEType',
    'PDEConfig',
    'HeatEquationSolver',
    'WaveEquationSolver',
    'ReactionDiffusionSolver',
    'AdvectionDiffusionSolver',
    'HamiltonJacobiSolver',
    'logistic_reaction',
    'competitive_reaction',
    'cubic_reaction',
    'create_heat_solver',
    'create_wave_solver',
    'create_reaction_diffusion_solver',
]


