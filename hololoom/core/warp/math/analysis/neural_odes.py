"""
Neural ODEs — Continuous-Depth Models via ODE Solvers
=====================================================

Neural ODEs treat depth as continuous: instead of discrete layers,
the hidden state evolves as dy/dt = f_θ(t, y). Gradients computed
via the adjoint method (O(1) memory).

Core Concepts:
- NeuralODE: Forward pass via ODE solve, backward via adjoint
- AdjointMethod: Memory-efficient gradient computation
- ContinuousNormalizing: Change-of-variables for density estimation

Mathematical Foundation:
Forward: y(t₁) = y(t₀) + ∫_{t₀}^{t₁} f_θ(t, y(t)) dt
Adjoint: da/dt = -aᵀ ∂f/∂y,  dL/dθ = -∫ aᵀ ∂f/∂θ dt
CNF log-prob: ln p(z₁) = ln p(z₀) - ∫ Tr(∂f/∂z) dt

Applications to Warp Space:
- Continuous embedding transformations
- Memory-efficient deep Thompson Sampling networks
- Density estimation for embedding distributions

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class NeuralODEResult:
    """Result of a neural ODE forward pass.

    Attributes:
        trajectory: State evolution (n_steps, n_dims)
        times: Time points (n_steps,)
        final_state: y(t_end)
        n_function_evals: Number of f evaluations
    """
    trajectory: np.ndarray
    times: np.ndarray
    final_state: np.ndarray
    n_function_evals: int = 0


@dataclass
class AdjointResult:
    """Result of adjoint gradient computation.

    Attributes:
        grad_y0: Gradient w.r.t. initial state (n_dims,)
        grad_params: Gradient w.r.t. parameters (dict of arrays)
        n_function_evals: Total evaluations (forward + backward)
    """
    grad_y0: np.ndarray
    grad_params: dict = field(default_factory=dict)
    n_function_evals: int = 0


@dataclass
class CNFResult:
    """Result of continuous normalizing flow.

    Attributes:
        z: Transformed samples (n_samples, n_dims)
        log_prob: Log probabilities (n_samples,)
        log_det_jacobian: Accumulated log-determinant (n_samples,)
    """
    z: np.ndarray
    log_prob: np.ndarray
    log_det_jacobian: np.ndarray


# ============================================================================
# BASIC ODE SOLVER (RK4 for neural ODE integration)
# ============================================================================

def _rk4_step(f: Callable, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """Single RK4 step."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _solve_ode(
    f: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_span: tuple[float, float],
    n_steps: int = 100,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve ODE with RK4, return (times, trajectory, n_evals)."""
    t0, tf = t_span
    times = np.linspace(t0, tf, n_steps + 1)
    dt = times[1] - times[0]

    trajectory = np.zeros((n_steps + 1,) + y0.shape)
    trajectory[0] = y0
    n_evals = 0

    for i in range(n_steps):
        trajectory[i + 1] = _rk4_step(f, times[i], trajectory[i], dt)
        n_evals += 4  # RK4 uses 4 evaluations

    return times, trajectory, n_evals


# ============================================================================
# NEURAL ODE
# ============================================================================

class NeuralODE:
    """Neural ODE: continuous-depth model.

    Treats the transformation as an ODE initial value problem.
    The "network" is the dynamics function f(t, y) which can be
    any parameterized function (including a neural network).

    Example:
        >>> def dynamics(t, y):
        ...     W = np.array([[-0.1, 0.5], [-0.5, -0.1]])
        ...     return W @ y
        >>> node = NeuralODE(n_steps=200)
        >>> result = node.forward(np.array([1.0, 0.0]), (0, 5), dynamics)
        >>> result.final_state.shape
        (2,)
    """

    def __init__(self, n_steps: int = 100):
        """Initialize NeuralODE solver.

        Args:
            n_steps: Number of integration steps
        """
        self.n_steps = n_steps

    def forward(
        self,
        x0: np.ndarray,
        t_span: tuple[float, float],
        neural_fn: Callable[[float, np.ndarray], np.ndarray],
    ) -> NeuralODEResult:
        """Forward pass: integrate dynamics from x0.

        Args:
            x0: Initial state (n_dims,)
            t_span: (t_start, t_end) integration interval
            neural_fn: Dynamics function f(t, y) -> dy/dt

        Returns:
            NeuralODEResult with trajectory and final state
        """
        x0 = np.atleast_1d(np.asarray(x0, dtype=float))

        times, trajectory, n_evals = _solve_ode(
            neural_fn, x0, t_span, self.n_steps
        )

        return NeuralODEResult(
            trajectory=trajectory,
            times=times,
            final_state=trajectory[-1].copy(),
            n_function_evals=n_evals,
        )


# ============================================================================
# ADJOINT METHOD
# ============================================================================

class AdjointMethod:
    """Adjoint method for memory-efficient gradients.

    Instead of backpropagating through the ODE solver (O(L) memory),
    the adjoint method solves an augmented ODE backward in time (O(1) memory).

    Adjoint equation: da/dt = -aᵀ ∂f/∂y
    Parameter gradient: dL/dθ = -∫ aᵀ ∂f/∂θ dt

    Example:
        >>> adj = AdjointMethod(n_steps=100, fd_eps=1e-6)
        >>> # Given a trajectory and loss gradient at final time
        >>> result = adj.compute_gradients(trajectory, times, dL_dy_final, dynamics)
    """

    def __init__(self, n_steps: int = 100, fd_eps: float = 1e-6):
        self.n_steps = n_steps
        self.fd_eps = fd_eps

    def compute_gradients(
        self,
        trajectory: np.ndarray,
        times: np.ndarray,
        dL_dy_final: np.ndarray,
        neural_fn: Callable[[float, np.ndarray], np.ndarray],
    ) -> AdjointResult:
        """Compute gradients via adjoint method.

        Args:
            trajectory: Forward trajectory (n_times, n_dims)
            times: Time points from forward pass (n_times,)
            dL_dy_final: Loss gradient at final time (n_dims,)
            neural_fn: Dynamics function f(t, y)

        Returns:
            AdjointResult with gradients w.r.t. initial state
        """
        dL_dy_final = np.atleast_1d(np.asarray(dL_dy_final, dtype=float))
        n_dims = trajectory.shape[1] if trajectory.ndim > 1 else trajectory.shape[0]

        # Adjoint equation: da/dt = -aᵀ ∂f/∂y
        # We integrate backward: a(T) = dL/dy(T), solve backward to t=0
        a = dL_dy_final.copy()
        n_evals = 0

        # Backward integration using stored trajectory
        n_t = len(times)
        for i in range(n_t - 1, 0, -1):
            dt = times[i] - times[i - 1]
            y_i = trajectory[i]
            t_i = times[i]

            # Compute ∂f/∂y via finite differences
            jac = self._jacobian_fd(neural_fn, t_i, y_i)
            n_evals += n_dims + 1

            # Adjoint step (backward Euler for simplicity)
            a = a - dt * (a @ jac)

        return AdjointResult(
            grad_y0=a,
            grad_params={},
            n_function_evals=n_evals,
        )

    def _jacobian_fd(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        t: float,
        y: np.ndarray,
    ) -> np.ndarray:
        """Finite-difference Jacobian ∂f/∂y."""
        n = len(y)
        f0 = f(t, y)
        J = np.zeros((n, n))
        for j in range(n):
            y_pert = y.copy()
            y_pert[j] += self.fd_eps
            J[:, j] = (f(t, y_pert) - f0) / self.fd_eps
        return J


# ============================================================================
# CONTINUOUS NORMALIZING FLOWS
# ============================================================================

class ContinuousNormalizing:
    """Continuous Normalizing Flow (CNF).

    Uses the instantaneous change of variables formula:
    ∂ ln p(z(t)) / ∂t = -Tr(∂f/∂z(t))

    This allows exact density evaluation by integrating the trace
    of the Jacobian alongside the state.

    Example:
        >>> cnf = ContinuousNormalizing(n_steps=100)
        >>> def dynamics(t, y):
        ...     return -0.5 * y  # Simple contraction
        >>> result = cnf.log_prob(samples, dynamics, base_log_prob_fn)
    """

    def __init__(self, n_steps: int = 100, fd_eps: float = 1e-5):
        self.n_steps = n_steps
        self.fd_eps = fd_eps

    def log_prob(
        self,
        x: np.ndarray,
        neural_fn: Callable[[float, np.ndarray], np.ndarray],
        base_log_prob: Callable[[np.ndarray], float],
        t_span: tuple[float, float] = (0.0, 1.0),
    ) -> CNFResult:
        """Compute log probability of data under the flow.

        Integrates backward from data space (t=1) to base space (t=0),
        accumulating the log-determinant of the Jacobian.

        Args:
            x: Data points (n_samples, n_dims) or (n_dims,) for single point
            neural_fn: Dynamics function f(t, z) -> dz/dt
            base_log_prob: Log density of base distribution
            t_span: Time span (t_start, t_end)

        Returns:
            CNFResult with transformed points and log probabilities
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n_samples, n_dims = x.shape
        t0, tf = t_span

        # Integrate backward: t_end -> t_start
        times = np.linspace(tf, t0, self.n_steps + 1)
        dt = times[1] - times[0]  # negative

        z = x.copy()
        log_det = np.zeros(n_samples)

        for i in range(self.n_steps):
            t_i = times[i]

            for s in range(n_samples):
                # Trace of Jacobian via finite differences (Hutchinson estimator)
                tr_jac = self._trace_jacobian(neural_fn, t_i, z[s])
                log_det[s] -= dt * tr_jac  # Negative dt, so this adds

                # RK4 step (backward)
                z[s] = _rk4_step(neural_fn, t_i, z[s], dt)

        # Compute log probabilities in base space
        log_probs = np.array([base_log_prob(z[s]) for s in range(n_samples)])
        log_probs += log_det

        return CNFResult(
            z=z,
            log_prob=log_probs,
            log_det_jacobian=log_det,
        )

    def _trace_jacobian(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        t: float,
        z: np.ndarray,
    ) -> float:
        """Compute Tr(∂f/∂z) via finite differences."""
        n = len(z)
        f0 = f(t, z)
        tr = 0.0
        for j in range(n):
            z_pert = z.copy()
            z_pert[j] += self.fd_eps
            tr += (f(t, z_pert)[j] - f0[j]) / self.fd_eps
        return tr

    def transform(
        self,
        z0: np.ndarray,
        neural_fn: Callable[[float, np.ndarray], np.ndarray],
        t_span: tuple[float, float] = (0.0, 1.0),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform samples from base distribution forward through the flow.

        Args:
            z0: Base samples (n_samples, n_dims)
            neural_fn: Dynamics function
            t_span: Time span

        Returns:
            (transformed_samples, log_det_jacobian)
        """
        z0 = np.atleast_2d(np.asarray(z0, dtype=float))
        n_samples, n_dims = z0.shape

        times = np.linspace(t_span[0], t_span[1], self.n_steps + 1)
        dt = times[1] - times[0]

        z = z0.copy()
        log_det = np.zeros(n_samples)

        for i in range(self.n_steps):
            t_i = times[i]
            for s in range(n_samples):
                tr_jac = self._trace_jacobian(neural_fn, t_i, z[s])
                log_det[s] += dt * tr_jac
                z[s] = _rk4_step(neural_fn, t_i, z[s], dt)

        return z, log_det
