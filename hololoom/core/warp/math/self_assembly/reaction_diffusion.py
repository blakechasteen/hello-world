"""
Reaction-Diffusion Systems — Turing Patterns, Gray-Scott, FitzHugh-Nagumo
==========================================================================

Pattern formation through the interplay of local reaction kinetics
and spatial diffusion. Turing's insight: diffusion-driven instability
can create stable spatial patterns from homogeneous states.

Core Concepts:
- Reaction-Diffusion: ∂u/∂t = D∇²u + f(u) — diffusion + local reactions
- Turing Instability: Diffusion destabilizes a stable homogeneous equilibrium
- Gray-Scott: Two-species model producing spots, stripes, and labyrinths
- FitzHugh-Nagumo: Excitable medium model (simplified Hodgkin-Huxley)

Mathematical Foundation:
General: ∂u/∂t = Du ∇²u + f(u,v),  ∂v/∂t = Dv ∇²v + g(u,v)
Turing condition: f_u + g_v < 0, f_u*g_v - f_v*g_u > 0,
                  Dv*f_u + Du*g_v > 0 (diffusion-driven instability)
Gray-Scott: f = -uv² + F(1-u),  g = uv² - (F+k)v
FHN: εv̇ = v - v³/3 - w + I,  ẇ = v + a - bw

Applications to Warp Space:
- Self-organizing pattern formation in embedding spaces
- Excitable dynamics for attention/activation propagation
- Turing patterns for spatial memory layout optimization
- Wave propagation models for information diffusion through graphs

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
class RDResult:
    """Result of reaction-diffusion simulation.

    Attributes:
        u_final: Final state of first species (ny, nx) or (n,).
        v_final: Final state of second species.
        u_history: Time history of u (sampled) (n_snapshots, ny, nx).
        v_history: Time history of v (sampled).
        time: Time values at snapshots.
        pattern_type: Detected pattern type ('spots', 'stripes', 'labyrinth', 'uniform').
    """
    u_final: np.ndarray
    v_final: np.ndarray
    u_history: np.ndarray | None = None
    v_history: np.ndarray | None = None
    time: np.ndarray | None = None
    pattern_type: str = 'unknown'


# ============================================================================
# GENERAL REACTION-DIFFUSION
# ============================================================================

class ReactionDiffusion:
    """General two-species reaction-diffusion system on a 2D grid.

    ∂u/∂t = Du ∇²u + f(u, v)
    ∂v/∂t = Dv ∇²v + g(u, v)

    Uses finite differences for the Laplacian and explicit Euler
    time stepping. Periodic boundary conditions.

    Example:
        >>> rd = ReactionDiffusion()
        >>> u0 = np.random.rand(64, 64) * 0.1
        >>> v0 = np.random.rand(64, 64) * 0.1
        >>> f = lambda u, v: u * (1 - u) - u * v  # Prey growth
        >>> g = lambda u, v: -0.5 * v + u * v      # Predator
        >>> result = rd.evolve(u0, v0, f, Du=0.01, Dv=0.005, dx=1.0, dt=0.01, n_steps=5000)
    """

    @staticmethod
    def _laplacian_2d(field: np.ndarray, dx: float) -> np.ndarray:
        """2D Laplacian via 5-point stencil with periodic BC.

        ∇²f ≈ (f_{i+1,j} + f_{i-1,j} + f_{i,j+1} + f_{i,j-1} - 4f_{i,j}) / dx²
        """
        return (
            np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0)
            + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)
            - 4.0 * field
        ) / (dx * dx)

    def evolve(
        self,
        u: np.ndarray,
        v: np.ndarray,
        f_reaction: Callable[[np.ndarray, np.ndarray], np.ndarray],
        Du: float = 0.01,
        Dv: float = 0.005,
        dx: float = 1.0,
        dt: float = 0.01,
        n_steps: int = 1000,
        save_interval: int = 100,
    ) -> RDResult:
        """Evolve the reaction-diffusion system.

        Args:
            u: Initial state of species u (ny, nx).
            v: Initial state of species v (ny, nx).
            f_reaction: Reaction term f(u, v) returning (du, dv) tuple
                       or just du (with g_reaction being -f_reaction for activator-inhibitor).
            Du: Diffusion coefficient for u.
            Dv: Diffusion coefficient for v.
            dx: Spatial step size.
            dt: Time step size.
            n_steps: Number of time steps.
            save_interval: Steps between snapshots.

        Returns:
            RDResult with final and historical states.
        """
        u = np.asarray(u, dtype=float).copy()
        v = np.asarray(v, dtype=float).copy()

        u_history = [u.copy()]
        v_history = [v.copy()]
        times = [0.0]

        for step in range(n_steps):
            lap_u = self._laplacian_2d(u, dx)
            lap_v = self._laplacian_2d(v, dx)

            # Reaction terms
            reaction = f_reaction(u, v)
            if isinstance(reaction, tuple):
                du_react, dv_react = reaction
            else:
                du_react = reaction
                dv_react = -reaction  # Default inhibitor

            u += dt * (Du * lap_u + du_react)
            v += dt * (Dv * lap_v + dv_react)

            if (step + 1) % save_interval == 0:
                u_history.append(u.copy())
                v_history.append(v.copy())
                times.append((step + 1) * dt)

        pattern = self._classify_pattern(u)

        return RDResult(
            u_final=u,
            v_final=v,
            u_history=np.array(u_history),
            v_history=np.array(v_history),
            time=np.array(times),
            pattern_type=pattern,
        )

    @staticmethod
    def _classify_pattern(field: np.ndarray) -> str:
        """Simple pattern classification via spatial frequency analysis."""
        if field.ndim < 2:
            return 'unknown'

        # Check if approximately uniform
        if np.std(field) < 0.01 * np.abs(np.mean(field) + 1e-10):
            return 'uniform'

        # FFT-based analysis
        fft = np.fft.fft2(field - np.mean(field))
        power = np.abs(fft) ** 2
        power[0, 0] = 0  # Remove DC

        # Find dominant frequency
        max_idx = np.unravel_index(np.argmax(power), power.shape)
        ny, nx = field.shape

        # Radial vs angular distribution of power
        yy, xx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing='ij')
        radial = np.sqrt(xx ** 2 + yy ** 2)
        angular = np.arctan2(yy, xx)

        # Spots: isotropic power at one radial frequency
        # Stripes: power concentrated at specific angles
        mean_r = np.average(radial, weights=power + 1e-10)
        if mean_r > 0:
            angular_var = np.average((angular - np.average(angular, weights=power + 1e-10)) ** 2,
                                     weights=power + 1e-10)
            if angular_var > 1.0:
                return 'spots'
            elif angular_var < 0.3:
                return 'stripes'

        return 'labyrinth'

    @staticmethod
    def turing_condition(
        fu: float, fv: float, gu: float, gv: float, Du: float, Dv: float
    ) -> bool:
        """Check Turing instability conditions.

        For the equilibrium to be Turing unstable, we need:
        1. Stable without diffusion: fu + gv < 0 and fu*gv - fv*gu > 0
        2. Diffusion-driven instability: Dv*fu + Du*gv > 0

        Args:
            fu, fv, gu, gv: Jacobian entries at equilibrium.
            Du, Dv: Diffusion coefficients.

        Returns:
            True if Turing instability conditions are met.
        """
        trace = fu + gv
        det = fu * gv - fv * gu
        diffusion_trace = Dv * fu + Du * gv
        discriminant = diffusion_trace ** 2 - 4 * Du * Dv * det

        return (trace < 0 and det > 0 and diffusion_trace > 0 and discriminant > 0)


# ============================================================================
# GRAY-SCOTT MODEL
# ============================================================================

class GrayScott:
    """Gray-Scott reaction-diffusion model.

    ∂u/∂t = Du ∇²u - uv² + F(1 - u)
    ∂v/∂t = Dv ∇²v + uv² - (F + k)v

    u: substrate concentration (fed at rate F)
    v: catalyst concentration (removed at rate F + k)

    Rich pattern zoo depending on (F, k):
    - F=0.04, k=0.06: spots
    - F=0.035, k=0.065: stripes
    - F=0.012, k=0.05: coral/worms
    - F=0.025, k=0.05: mitosis (dividing spots)

    Example:
        >>> gs = GrayScott()
        >>> u0, v0 = gs.initial_condition(128, seed_type='square')
        >>> result = gs.evolve(u0, v0, F=0.04, k=0.06, n_steps=10000)
    """

    def initial_condition(
        self, size: int = 64, seed_type: str = 'square'
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate initial conditions.

        Args:
            size: Grid size (size x size).
            seed_type: 'square', 'random', or 'circle'.

        Returns:
            (u0, v0) initial states.
        """
        u = np.ones((size, size))
        v = np.zeros((size, size))

        center = size // 2
        r = size // 10

        if seed_type == 'square':
            u[center - r: center + r, center - r: center + r] = 0.5
            v[center - r: center + r, center - r: center + r] = 0.25
        elif seed_type == 'circle':
            yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
            dist = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
            mask = dist < r
            u[mask] = 0.5
            v[mask] = 0.25
        elif seed_type == 'random':
            u += 0.02 * np.random.randn(size, size)
            v += 0.01 * np.random.randn(size, size)
            v = np.maximum(v, 0)

        # Add small noise for symmetry breaking
        u += 0.01 * np.random.randn(size, size)
        v += 0.01 * np.random.randn(size, size)
        v = np.maximum(v, 0)

        return u, v

    def evolve(
        self,
        u: np.ndarray,
        v: np.ndarray,
        F: float = 0.04,
        k: float = 0.06,
        Du: float = 0.16,
        Dv: float = 0.08,
        dx: float = 1.0,
        dt: float = 1.0,
        n_steps: int = 1000,
        save_interval: int = 100,
    ) -> RDResult:
        """Evolve Gray-Scott system.

        Args:
            u, v: Initial states (ny, nx).
            F: Feed rate.
            k: Kill rate.
            Du: Diffusion coefficient for u.
            Dv: Diffusion coefficient for v.
            dx: Spatial step.
            dt: Time step.
            n_steps: Total steps.
            save_interval: Steps between snapshots.

        Returns:
            RDResult with evolution data.
        """
        u = u.copy()
        v = v.copy()
        u_hist = [u.copy()]
        v_hist = [v.copy()]
        times = [0.0]

        for step in range(n_steps):
            lap_u = ReactionDiffusion._laplacian_2d(u, dx)
            lap_v = ReactionDiffusion._laplacian_2d(v, dx)

            uv2 = u * v * v
            u += dt * (Du * lap_u - uv2 + F * (1.0 - u))
            v += dt * (Dv * lap_v + uv2 - (F + k) * v)

            # Clamp to physical range
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)

            if (step + 1) % save_interval == 0:
                u_hist.append(u.copy())
                v_hist.append(v.copy())
                times.append((step + 1) * dt)

        pattern = ReactionDiffusion._classify_pattern(v)

        return RDResult(
            u_final=u,
            v_final=v,
            u_history=np.array(u_hist),
            v_history=np.array(v_hist),
            time=np.array(times),
            pattern_type=pattern,
        )


# ============================================================================
# FITZHUGH-NAGUMO MODEL
# ============================================================================

class FitzHughNagumo:
    """FitzHugh-Nagumo excitable medium model.

    ε ∂v/∂t = Dv ∇²v + v - v³/3 - w + I_ext
    ∂w/∂t = Dw ∇²w + v + a - b*w

    v: fast "voltage" variable (excitation)
    w: slow "recovery" variable
    ε: time scale separation (small ε = fast excitation)

    Produces traveling waves, spiral waves, and excitable pulses.

    Example:
        >>> fhn = FitzHughNagumo()
        >>> v0 = np.random.randn(64, 64) * 0.1
        >>> w0 = np.zeros((64, 64))
        >>> result = fhn.evolve(v0, w0, I_ext=0.5, n_steps=5000)
    """

    def evolve(
        self,
        v: np.ndarray,
        w: np.ndarray,
        I_ext: float = 0.0,
        a: float = 0.7,
        b: float = 0.8,
        tau: float = 12.5,
        epsilon: float = 0.08,
        Dv: float = 1.0,
        Dw: float = 0.0,
        dx: float = 1.0,
        dt: float = 0.01,
        n_steps: int = 1000,
        save_interval: int = 100,
    ) -> RDResult:
        """Evolve FitzHugh-Nagumo system.

        Args:
            v: Initial voltage field (ny, nx) or (n,).
            w: Initial recovery field.
            I_ext: External current (scalar or array).
            a: Recovery parameter.
            b: Recovery coupling.
            tau: Recovery time constant.
            epsilon: Time scale separation.
            Dv: Voltage diffusion.
            Dw: Recovery diffusion (usually 0 or small).
            dx: Spatial step.
            dt: Time step.
            n_steps: Total steps.
            save_interval: Steps between snapshots.

        Returns:
            RDResult with evolution data.
        """
        v = np.asarray(v, dtype=float).copy()
        w = np.asarray(w, dtype=float).copy()
        is_2d = v.ndim == 2

        v_hist = [v.copy()]
        w_hist = [w.copy()]
        times = [0.0]

        for step in range(n_steps):
            if is_2d:
                lap_v = ReactionDiffusion._laplacian_2d(v, dx)
                lap_w = ReactionDiffusion._laplacian_2d(w, dx) if Dw > 0 else 0.0
            else:
                # 1D Laplacian
                lap_v = np.zeros_like(v)
                lap_v[1:-1] = (v[2:] - 2 * v[1:-1] + v[:-2]) / (dx * dx)
                lap_v[0] = (v[1] - 2 * v[0] + v[-1]) / (dx * dx)
                lap_v[-1] = (v[0] - 2 * v[-1] + v[-2]) / (dx * dx)
                lap_w = 0.0

            # FHN dynamics
            dv = (Dv * lap_v + v - v ** 3 / 3.0 - w + I_ext) / epsilon
            dw = (v + a - b * w) / tau
            if Dw > 0:
                dw += Dw * lap_w / tau

            v += dt * dv
            w += dt * dw

            if (step + 1) % save_interval == 0:
                v_hist.append(v.copy())
                w_hist.append(w.copy())
                times.append((step + 1) * dt)

        pattern = 'excitable'
        if is_2d:
            # Check for spiral waves
            v_range = np.max(v) - np.min(v)
            if v_range > 1.0:
                pattern = 'spiral_wave'
            elif v_range < 0.1:
                pattern = 'quiescent'

        return RDResult(
            u_final=v,
            v_final=w,
            u_history=np.array(v_hist),
            v_history=np.array(w_hist),
            time=np.array(times),
            pattern_type=pattern,
        )

    @staticmethod
    def nullclines(a: float = 0.7, b: float = 0.8, I_ext: float = 0.0,
                   v_range: tuple[float, float] = (-2.5, 2.5), n_points: int = 200
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute v- and w-nullclines for phase plane analysis.

        v-nullcline: w = v - v³/3 + I_ext
        w-nullcline: w = (v + a) / b

        Args:
            a, b, I_ext: FHN parameters.
            v_range: Range of v values.
            n_points: Number of points.

        Returns:
            (v_values, w_v_nullcline, w_w_nullcline).
        """
        v = np.linspace(v_range[0], v_range[1], n_points)
        w_vnull = v - v ** 3 / 3.0 + I_ext
        w_wnull = (v + a) / b
        return v, w_vnull, w_wnull
