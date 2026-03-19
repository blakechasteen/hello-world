"""
Optimal Control — Pontryagin, LQR, Model Predictive Control
=============================================================

Methods for finding control inputs that optimize a cost functional
subject to dynamical system constraints.

Core Concepts:
- Pontryagin Maximum Principle: Necessary conditions via costate/Hamiltonian
- LQR: Closed-form for linear dynamics + quadratic cost
- MPC: Receding-horizon optimization for constrained systems

Mathematical Foundation:
Pontryagin: H(x, u, λ) = L(x, u) + λᵀf(x, u),  λ̇ = -∂H/∂x
LQR: K = R⁻¹BᵀP where P solves Pₐ = AᵀPA - AᵀPB(R+BᵀPB)⁻¹BᵀPA + Q
MPC: min_{u₀..u_{N-1}} Σ l(xₖ, uₖ) + V(x_N) s.t. xₖ₊₁ = f(xₖ, uₖ)

Applications to Warp Space:
- Optimal scheduling of weaving cycle resources
- LQR for embedding trajectory stabilization
- MPC for online Thompson Sampling budget allocation

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.linalg import solve_continuous_are, solve_discrete_are
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ControlResult:
    """Result of optimal control computation.

    Attributes:
        states: State trajectory (n_steps+1, n_states)
        controls: Control inputs (n_steps, n_controls)
        costates: Costate/adjoint trajectory (n_steps+1, n_states)
        times: Time points (n_steps+1,)
        cost: Total cost
        converged: Whether the method converged
    """
    states: np.ndarray
    controls: np.ndarray
    costates: np.ndarray = field(default_factory=lambda: np.empty(0))
    times: np.ndarray = field(default_factory=lambda: np.empty(0))
    cost: float = 0.0
    converged: bool = True


@dataclass
class LQRResult:
    """Result of LQR computation.

    Attributes:
        K: Feedback gain matrix (n_controls, n_states)
        P: Solution to Riccati equation (n_states, n_states)
        eigenvalues: Closed-loop eigenvalues
    """
    K: np.ndarray
    P: np.ndarray
    eigenvalues: np.ndarray = field(default_factory=lambda: np.empty(0))


@dataclass
class MPCResult:
    """Result of MPC computation.

    Attributes:
        action: Optimal first action u₀ (n_controls,)
        planned_states: Planned state trajectory (horizon+1, n_states)
        planned_actions: Full planned action sequence (horizon, n_controls)
        cost: Predicted total cost over horizon
    """
    action: np.ndarray
    planned_states: np.ndarray
    planned_actions: np.ndarray
    cost: float = 0.0


# ============================================================================
# PONTRYAGIN MAXIMUM PRINCIPLE
# ============================================================================

class PontryaginMaximum:
    """Solve optimal control via Pontryagin's Maximum Principle.

    Uses the indirect method: solve the two-point boundary value
    problem formed by state + costate equations.

    Hamiltonian: H(x, u, λ) = L(x, u) + λᵀf(x, u)
    State: ẋ = ∂H/∂λ = f(x, u)
    Costate: λ̇ = -∂H/∂x
    Optimality: ∂H/∂u = 0

    Example:
        >>> pmp = PontryaginMaximum(n_steps=200)
        >>> def dynamics(x, u): return np.array([x[1], u[0]])
        >>> def cost(x, u): return 0.5 * (x[0]**2 + u[0]**2)
        >>> def costate_dynamics(x, lam, u): return np.array([-lam[0], -lam[1]])
        >>> result = pmp.solve(dynamics, cost, costate_dynamics,
        ...                    (0, 2), np.array([1, 0]))
    """

    def __init__(self, n_steps: int = 200, max_iter: int = 50, tol: float = 1e-6):
        self.n_steps = n_steps
        self.max_iter = max_iter
        self.tol = tol

    def solve(
        self,
        dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray],
        cost: Callable[[np.ndarray, np.ndarray], float],
        costate_dynamics: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        t_span: tuple[float, float],
        x0: np.ndarray,
        control_from_costate: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        n_controls: int = 1,
    ) -> ControlResult:
        """Solve via forward-backward sweep (indirect shooting).

        Args:
            dynamics: ẋ = f(x, u)
            cost: Running cost L(x, u)
            costate_dynamics: λ̇ = g(x, λ, u), typically -∂H/∂x
            t_span: (t_start, t_end)
            x0: Initial state
            control_from_costate: u*(x, λ) from ∂H/∂u = 0 (optional)
            n_controls: Number of control inputs

        Returns:
            ControlResult with optimal trajectory
        """
        x0 = np.atleast_1d(np.asarray(x0, dtype=float))
        n_states = len(x0)
        t0, tf = t_span
        times = np.linspace(t0, tf, self.n_steps + 1)
        dt = times[1] - times[0]

        # Initialize
        states = np.zeros((self.n_steps + 1, n_states))
        costates = np.zeros((self.n_steps + 1, n_states))
        controls = np.zeros((self.n_steps, n_controls))

        states[0] = x0
        converged = False

        for iteration in range(self.max_iter):
            # Forward sweep: integrate state equations
            for k in range(self.n_steps):
                if control_from_costate is not None:
                    controls[k] = control_from_costate(states[k], costates[k])
                u_k = controls[k]
                dx = dynamics(states[k], u_k)
                states[k + 1] = states[k] + dt * dx

            # Backward sweep: integrate costate equations
            costates[-1] = np.zeros(n_states)  # Terminal condition
            old_costates = costates.copy()

            for k in range(self.n_steps - 1, -1, -1):
                if control_from_costate is not None:
                    controls[k] = control_from_costate(states[k], costates[k + 1])
                u_k = controls[k]
                dlam = costate_dynamics(states[k], costates[k + 1], u_k)
                costates[k] = costates[k + 1] - dt * dlam

            # Check convergence
            diff = np.max(np.abs(costates - old_costates))
            if diff < self.tol:
                converged = True
                break

        # Compute total cost
        total_cost = 0.0
        for k in range(self.n_steps):
            total_cost += dt * cost(states[k], controls[k])

        return ControlResult(
            states=states,
            controls=controls,
            costates=costates,
            times=times,
            cost=total_cost,
            converged=converged,
        )


# ============================================================================
# LINEAR QUADRATIC REGULATOR (LQR)
# ============================================================================

class LQR:
    """Linear-Quadratic Regulator via algebraic Riccati equation.

    For linear dynamics ẋ = Ax + Bu and quadratic cost
    J = ∫ (xᵀQx + uᵀRu) dt, the optimal control is u = -Kx
    where K = R⁻¹BᵀP and P solves the Riccati equation.

    Example:
        >>> lqr = LQR()
        >>> A = np.array([[0, 1], [0, 0]])  # Double integrator
        >>> B = np.array([[0], [1]])
        >>> Q = np.eye(2)
        >>> R = np.array([[1.0]])
        >>> result = lqr.solve(A, B, Q, R)
        >>> result.K.shape
        (1, 2)
    """

    def solve(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        continuous: bool = True,
    ) -> LQRResult:
        """Solve the LQR problem.

        Args:
            A: State matrix (n, n)
            B: Input matrix (n, m)
            Q: State cost matrix (n, n), positive semi-definite
            R: Control cost matrix (m, m), positive definite
            continuous: If True, solve continuous-time ARE; else discrete

        Returns:
            LQRResult with gain matrix K and Riccati solution P
        """
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        Q = np.asarray(Q, dtype=float)
        R = np.asarray(R, dtype=float)

        if HAS_SCIPY:
            if continuous:
                P = solve_continuous_are(A, B, Q, R)
            else:
                P = solve_discrete_are(A, B, Q, R)
        else:
            # Iterative solution to DARE/CARE
            P = self._solve_riccati_iterative(A, B, Q, R, continuous)

        # Compute gain
        R_inv = np.linalg.inv(R)
        if continuous:
            K = R_inv @ B.T @ P
        else:
            K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

        # Closed-loop eigenvalues
        A_cl = A - B @ K
        eigenvalues = np.linalg.eigvals(A_cl)

        return LQRResult(K=K, P=P, eigenvalues=eigenvalues)

    @staticmethod
    def _solve_riccati_iterative(
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        continuous: bool,
        max_iter: int = 1000,
        tol: float = 1e-10,
    ) -> np.ndarray:
        """Iterative Riccati solver (fallback without scipy)."""
        n = A.shape[0]
        P = Q.copy()
        R_inv = np.linalg.inv(R)

        for _ in range(max_iter):
            if continuous:
                # CARE iteration: P_{k+1} = Q + AᵀP_k + P_kA - P_kBR⁻¹BᵀP_k
                P_new = Q + A.T @ P + P @ A - P @ B @ R_inv @ B.T @ P
            else:
                # DARE iteration
                K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
                P_new = Q + A.T @ P @ A - A.T @ P @ B @ K

            if np.max(np.abs(P_new - P)) < tol:
                return P_new
            P = P_new

        logger.warning("Riccati iteration did not converge")
        return P


# ============================================================================
# MODEL PREDICTIVE CONTROL
# ============================================================================

class ModelPredictiveControl:
    """Model Predictive Control with receding horizon.

    At each step, solve a finite-horizon optimal control problem
    and apply only the first control action. Handles constraints
    via projected gradient.

    Example:
        >>> mpc = ModelPredictiveControl(horizon=20)
        >>> def dynamics(x, u): return A @ x + B @ u
        >>> def cost(x, u): return x @ Q @ x + u @ R @ u
        >>> result = mpc.solve(dynamics, cost, None, 20, x0)
    """

    def __init__(self, horizon: int = 20, n_opt_iter: int = 50, lr: float = 0.01):
        self.horizon = horizon
        self.n_opt_iter = n_opt_iter
        self.lr = lr

    def solve(
        self,
        dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray],
        cost: Callable[[np.ndarray, np.ndarray], float],
        constraints: tuple[np.ndarray, np.ndarray] | None = None,
        horizon: int | None = None,
        x0: np.ndarray | None = None,
        n_controls: int = 1,
        terminal_cost: Callable[[np.ndarray], float] | None = None,
    ) -> MPCResult:
        """Solve MPC problem via gradient descent on control sequence.

        Args:
            dynamics: xₖ₊₁ = f(xₖ, uₖ)
            cost: Stage cost l(x, u)
            constraints: (u_min, u_max) control bounds
            horizon: Planning horizon (overrides init)
            x0: Current state
            n_controls: Dimension of control input
            terminal_cost: Terminal cost V(x_N) (optional)

        Returns:
            MPCResult with optimal first action and planned trajectory
        """
        if x0 is None:
            raise ValueError("Initial state x0 required")

        x0 = np.atleast_1d(np.asarray(x0, dtype=float))
        H = horizon if horizon is not None else self.horizon
        n_states = len(x0)

        # Initialize control sequence
        u_seq = np.zeros((H, n_controls))

        # Gradient descent on control sequence
        fd_eps = 1e-5

        for opt_iter in range(self.n_opt_iter):
            # Forward rollout
            states, total_cost = self._rollout(
                dynamics, cost, x0, u_seq, terminal_cost
            )

            # Compute gradient via finite differences
            grad = np.zeros_like(u_seq)
            for k in range(H):
                for j in range(n_controls):
                    u_pert = u_seq.copy()
                    u_pert[k, j] += fd_eps
                    _, cost_pert = self._rollout(
                        dynamics, cost, x0, u_pert, terminal_cost
                    )
                    grad[k, j] = (cost_pert - total_cost) / fd_eps

            # Update
            u_seq -= self.lr * grad

            # Project onto constraints
            if constraints is not None:
                u_min, u_max = constraints
                u_seq = np.clip(u_seq, u_min, u_max)

        # Final rollout
        states, total_cost = self._rollout(
            dynamics, cost, x0, u_seq, terminal_cost
        )

        return MPCResult(
            action=u_seq[0].copy(),
            planned_states=states,
            planned_actions=u_seq,
            cost=total_cost,
        )

    @staticmethod
    def _rollout(
        dynamics: Callable,
        cost: Callable,
        x0: np.ndarray,
        u_seq: np.ndarray,
        terminal_cost: Callable | None = None,
    ) -> tuple[np.ndarray, float]:
        """Forward rollout of dynamics with control sequence."""
        H = len(u_seq)
        states = np.zeros((H + 1, len(x0)))
        states[0] = x0

        total_cost = 0.0
        for k in range(H):
            total_cost += cost(states[k], u_seq[k])
            states[k + 1] = dynamics(states[k], u_seq[k])

        if terminal_cost is not None:
            total_cost += terminal_cost(states[-1])

        return states, total_cost
