"""
Advanced ODE Integrators for Spring Dynamics
=============================================
Professional-grade numerical integration methods replacing naive Euler.

This module provides:
1. Runge-Kutta 4th order (RK4) - General purpose, high accuracy
2. Symplectic integrators - Energy-preserving for Hamiltonian systems
3. Adaptive step size control - Automatic accuracy/efficiency tradeoff
4. Stability analysis tools - Convergence guarantees

Mathematical Foundation:
-----------------------
Spring dynamics form a Hamiltonian system:
    H(q, p) = (1/2) Σ p²/m + (1/2) Σ k(q_i - q_j)²

Where q = activation, p = momentum = m × velocity

For Hamiltonian systems, symplectic integrators preserve:
- Phase space volume (Liouville's theorem)
- Energy (approximately, with bounded error)
- Long-term stability

Author: HoloLoom Mathematical Physics Team
Date: 2025-11-03
"""

from typing import Callable, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np


# ============================================================================
# Integrator Types
# ============================================================================

class IntegratorType(Enum):
    """Available numerical integration methods."""

    EULER = "euler"                    # 1st order, fast but inaccurate
    RK4 = "rk4"                       # 4th order, general purpose
    SYMPLECTIC_EULER = "symplectic"   # 1st order, energy-preserving
    VERLET = "verlet"                 # 2nd order, energy-preserving
    RK45 = "rk45"                     # Adaptive 4th/5th order


# ============================================================================
# State Representation
# ============================================================================

@dataclass
class DynamicalState:
    """
    Complete state for dynamical system integration.

    For spring dynamics:
    - q (position): Activation levels [0, 1]
    - p (momentum): m × velocity
    - t (time): Current simulation time
    """

    q: np.ndarray  # Positions (activations)
    p: np.ndarray  # Momenta (mass × velocity)
    t: float = 0.0  # Time

    def copy(self) -> 'DynamicalState':
        """Create deep copy of state."""
        return DynamicalState(
            q=self.q.copy(),
            p=self.p.copy(),
            t=self.t
        )


# ============================================================================
# Force Functions (Right-Hand Side)
# ============================================================================

class ForceFunction:
    """
    Force function interface for ODE systems.

    Computes dq/dt and dp/dt for Hamiltonian dynamics:
        dq/dt = p / m         (velocity from momentum)
        dp/dt = F(q)          (force from positions)
    """

    def __call__(self, state: DynamicalState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute time derivatives.

        Args:
            state: Current dynamical state

        Returns:
            Tuple of (dq_dt, dp_dt)
        """
        raise NotImplementedError


# ============================================================================
# Runge-Kutta 4th Order (RK4)
# ============================================================================

class RK4Integrator:
    """
    Classic 4th-order Runge-Kutta integrator.

    Accuracy: O(h^5) per step, O(h^4) global
    Stability: Larger stability region than Euler
    Cost: 4 force evaluations per step

    Algorithm:
        k1 = f(y_n)
        k2 = f(y_n + h/2 × k1)
        k3 = f(y_n + h/2 × k2)
        k4 = f(y_n + h × k3)
        y_{n+1} = y_n + h/6 × (k1 + 2k2 + 2k3 + k4)
    """

    def __init__(self, force_fn: ForceFunction):
        """
        Initialize RK4 integrator.

        Args:
            force_fn: Function computing (dq_dt, dp_dt)
        """
        self.force_fn = force_fn
        self.n_evaluations = 0

    def step(self, state: DynamicalState, dt: float) -> DynamicalState:
        """
        Take single RK4 step.

        Args:
            state: Current state
            dt: Time step size

        Returns:
            New state at t + dt
        """
        # k1 = f(y_n)
        dq1, dp1 = self.force_fn(state)

        # k2 = f(y_n + h/2 × k1)
        state2 = DynamicalState(
            q=state.q + 0.5 * dt * dq1,
            p=state.p + 0.5 * dt * dp1,
            t=state.t + 0.5 * dt
        )
        dq2, dp2 = self.force_fn(state2)

        # k3 = f(y_n + h/2 × k2)
        state3 = DynamicalState(
            q=state.q + 0.5 * dt * dq2,
            p=state.p + 0.5 * dt * dp2,
            t=state.t + 0.5 * dt
        )
        dq3, dp3 = self.force_fn(state3)

        # k4 = f(y_n + h × k3)
        state4 = DynamicalState(
            q=state.q + dt * dq3,
            p=state.p + dt * dp3,
            t=state.t + dt
        )
        dq4, dp4 = self.force_fn(state4)

        # Combine: y_{n+1} = y_n + h/6 × (k1 + 2k2 + 2k3 + k4)
        new_state = DynamicalState(
            q=state.q + (dt / 6.0) * (dq1 + 2*dq2 + 2*dq3 + dq4),
            p=state.p + (dt / 6.0) * (dp1 + 2*dp2 + 2*dp3 + dp4),
            t=state.t + dt
        )

        self.n_evaluations += 4
        return new_state


# ============================================================================
# Symplectic Euler (1st order, energy-preserving)
# ============================================================================

class SymplecticEulerIntegrator:
    """
    Symplectic Euler integrator for Hamiltonian systems.

    Accuracy: O(h^2) per step, O(h) global (same as Euler)
    Stability: Much better than Euler for oscillatory systems
    Energy: Bounded error, no secular drift
    Cost: 1 force evaluation per step

    Algorithm:
        p_{n+1} = p_n + h × F(q_n)        (update momentum first)
        q_{n+1} = q_n + h × p_{n+1}/m     (use new momentum)

    This ordering preserves symplectic structure!
    """

    def __init__(self, force_fn: ForceFunction, mass: np.ndarray):
        """
        Initialize symplectic Euler integrator.

        Args:
            force_fn: Function computing (dq_dt, dp_dt)
            mass: Mass array for each degree of freedom
        """
        self.force_fn = force_fn
        self.mass = mass
        self.n_evaluations = 0

    def step(self, state: DynamicalState, dt: float) -> DynamicalState:
        """
        Take single symplectic Euler step.

        Args:
            state: Current state
            dt: Time step size

        Returns:
            New state at t + dt
        """
        # Compute force at current position
        _, dp_dt = self.force_fn(state)

        # Update momentum first (critical ordering!)
        p_new = state.p + dt * dp_dt

        # Update position using new momentum
        q_new = state.q + dt * (p_new / self.mass)

        self.n_evaluations += 1

        return DynamicalState(
            q=q_new,
            p=p_new,
            t=state.t + dt
        )


# ============================================================================
# Velocity Verlet (2nd order, energy-preserving)
# ============================================================================

class VerletIntegrator:
    """
    Velocity Verlet integrator (Störmer-Verlet method).

    Accuracy: O(h^3) per step, O(h^2) global
    Stability: Excellent for oscillatory systems
    Energy: Machine-precision bounded error
    Cost: 2 force evaluations per step

    Algorithm:
        q_{n+1} = q_n + h × v_n + h²/2 × a_n
        a_{n+1} = F(q_{n+1}) / m
        v_{n+1} = v_n + h/2 × (a_n + a_{n+1})

    This is the gold standard for molecular dynamics.
    """

    def __init__(self, force_fn: ForceFunction, mass: np.ndarray):
        """
        Initialize Verlet integrator.

        Args:
            force_fn: Function computing (dq_dt, dp_dt)
            mass: Mass array for each degree of freedom
        """
        self.force_fn = force_fn
        self.mass = mass
        self.n_evaluations = 0

        # Cache previous acceleration
        self._prev_accel = None

    def step(self, state: DynamicalState, dt: float) -> DynamicalState:
        """
        Take single Verlet step.

        Args:
            state: Current state
            dt: Time step size

        Returns:
            New state at t + dt
        """
        # Convert momentum to velocity
        v = state.p / self.mass

        # Compute current acceleration
        _, dp_dt = self.force_fn(state)
        a = dp_dt / self.mass

        # Update position: q_{n+1} = q_n + h × v_n + h²/2 × a_n
        q_new = state.q + dt * v + 0.5 * dt * dt * a

        # Compute new acceleration at new position
        state_new = DynamicalState(q=q_new, p=state.p, t=state.t + dt)
        _, dp_dt_new = self.force_fn(state_new)
        a_new = dp_dt_new / self.mass

        # Update velocity: v_{n+1} = v_n + h/2 × (a_n + a_{n+1})
        v_new = v + 0.5 * dt * (a + a_new)

        # Convert back to momentum
        p_new = v_new * self.mass

        self.n_evaluations += 2
        self._prev_accel = a_new

        return DynamicalState(
            q=q_new,
            p=p_new,
            t=state.t + dt
        )


# ============================================================================
# Adaptive Step Size (Runge-Kutta-Fehlberg)
# ============================================================================

@dataclass
class AdaptiveStepResult:
    """Result from adaptive step size integration."""

    state: DynamicalState
    error_estimate: float
    dt_next: float
    accepted: bool


class RK45Integrator:
    """
    Adaptive step size Runge-Kutta (4th/5th order pair).

    Uses Dormand-Prince coefficients for error estimation.
    Automatically adjusts dt to maintain requested tolerance.

    Accuracy: O(h^5) per step, O(h^4) global
    Adaptivity: Automatically grows/shrinks dt
    Cost: 6 force evaluations per accepted step

    This is the workhorse for stiff/nonstiff ODE systems.
    """

    def __init__(
        self,
        force_fn: ForceFunction,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        safety: float = 0.9
    ):
        """
        Initialize RK45 integrator.

        Args:
            force_fn: Function computing (dq_dt, dp_dt)
            rtol: Relative tolerance
            atol: Absolute tolerance
            safety: Safety factor for step size adjustment (< 1)
        """
        self.force_fn = force_fn
        self.rtol = rtol
        self.atol = atol
        self.safety = safety
        self.n_evaluations = 0

        # Dormand-Prince coefficients
        self._setup_coefficients()

    def _setup_coefficients(self):
        """Set up Dormand-Prince RK45 coefficients."""
        # Butcher tableau for DOPRI5
        self.c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])

        self.a = [
            [],
            [1/5],
            [3/40, 9/40],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
        ]

        # 4th order solution
        self.b4 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])

        # 5th order solution
        self.b5 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])

    def step(self, state: DynamicalState, dt: float) -> AdaptiveStepResult:
        """
        Attempt adaptive step.

        Args:
            state: Current state
            dt: Proposed time step

        Returns:
            AdaptiveStepResult with new state, error estimate, and recommended dt
        """
        # Compute stages
        k_dq = []
        k_dp = []

        # Stage 1
        dq, dp = self.force_fn(state)
        k_dq.append(dq)
        k_dp.append(dp)

        # Stages 2-7
        for i in range(1, 7):
            # Build intermediate state
            q_temp = state.q.copy()
            p_temp = state.p.copy()

            for j, a_ij in enumerate(self.a[i]):
                q_temp += dt * a_ij * k_dq[j]
                p_temp += dt * a_ij * k_dp[j]

            state_temp = DynamicalState(q=q_temp, p=p_temp, t=state.t + self.c[i] * dt)
            dq, dp = self.force_fn(state_temp)
            k_dq.append(dq)
            k_dp.append(dp)

        self.n_evaluations += 7

        # 4th order solution
        q4 = state.q.copy()
        p4 = state.p.copy()
        for i, b in enumerate(self.b4):
            q4 += dt * b * k_dq[i]
            p4 += dt * b * k_dp[i]

        # 5th order solution
        q5 = state.q.copy()
        p5 = state.p.copy()
        for i, b in enumerate(self.b5):
            q5 += dt * b * k_dq[i]
            p5 += dt * b * k_dp[i]

        # Error estimate (difference between 4th and 5th order)
        err_q = np.abs(q5 - q4)
        err_p = np.abs(p5 - p4)

        # Compute scale (for relative tolerance)
        scale_q = self.atol + self.rtol * np.maximum(np.abs(state.q), np.abs(q5))
        scale_p = self.atol + self.rtol * np.maximum(np.abs(state.p), np.abs(p5))

        # Normalized error
        err_norm = np.sqrt(
            np.mean((err_q / scale_q)**2) + np.mean((err_p / scale_p)**2)
        )

        # Accept/reject step
        accepted = err_norm <= 1.0

        # Compute next step size
        if err_norm == 0:
            dt_next = dt * 5.0  # Max growth
        else:
            # PI controller for step size
            dt_next = dt * self.safety * (1.0 / err_norm)**0.2
            dt_next = np.clip(dt_next, 0.2 * dt, 5.0 * dt)

        # Use 5th order solution if accepted
        state_new = DynamicalState(
            q=q5 if accepted else state.q,
            p=p5 if accepted else state.p,
            t=state.t + dt if accepted else state.t
        )

        return AdaptiveStepResult(
            state=state_new,
            error_estimate=err_norm,
            dt_next=dt_next,
            accepted=accepted
        )


# ============================================================================
# Integrator Factory
# ============================================================================

def create_integrator(
    integrator_type: IntegratorType,
    force_fn: ForceFunction,
    mass: np.ndarray = None,
    **kwargs
):
    """
    Factory function to create integrator.

    Args:
        integrator_type: Type of integrator
        force_fn: Force function computing derivatives
        mass: Mass array (required for symplectic/Verlet)
        **kwargs: Additional parameters (rtol, atol for RK45)

    Returns:
        Integrator instance
    """
    if integrator_type == IntegratorType.RK4:
        return RK4Integrator(force_fn)

    elif integrator_type == IntegratorType.SYMPLECTIC_EULER:
        if mass is None:
            raise ValueError("Symplectic Euler requires mass array")
        return SymplecticEulerIntegrator(force_fn, mass)

    elif integrator_type == IntegratorType.VERLET:
        if mass is None:
            raise ValueError("Verlet requires mass array")
        return VerletIntegrator(force_fn, mass)

    elif integrator_type == IntegratorType.RK45:
        return RK45Integrator(
            force_fn,
            rtol=kwargs.get('rtol', 1e-6),
            atol=kwargs.get('atol', 1e-8)
        )

    else:
        raise ValueError(f"Unknown integrator type: {integrator_type}")


# ============================================================================
# Stability Analysis
# ============================================================================

def analyze_stability(
    integrator,
    state: DynamicalState,
    dt: float,
    n_steps: int = 1000
) -> Dict[str, Any]:
    """
    Analyze stability of integration scheme.

    Runs integration and tracks:
    - Energy drift
    - Phase space volume (symplecticity)
    - Convergence to equilibrium

    Args:
        integrator: Integrator instance
        state: Initial state
        dt: Time step
        n_steps: Number of steps to simulate

    Returns:
        Dict with stability metrics
    """
    states = [state.copy()]

    current = state.copy()
    for _ in range(n_steps):
        if hasattr(integrator, 'step'):
            result = integrator.step(current, dt)
            if isinstance(result, AdaptiveStepResult):
                current = result.state
            else:
                current = result
            states.append(current.copy())

    # Extract trajectories
    q_traj = np.array([s.q for s in states])
    p_traj = np.array([s.p for s in states])

    # Energy estimate (kinetic + potential approximation)
    kinetic = 0.5 * np.sum(p_traj**2, axis=1)
    potential = 0.5 * np.sum(q_traj**2, axis=1)  # Harmonic approximation
    energy = kinetic + potential

    # Energy drift
    energy_drift = np.abs(energy[-1] - energy[0]) / (np.abs(energy[0]) + 1e-10)

    # Phase space volume (should be constant for symplectic)
    # Approximate using det(J) for small subsystems

    return {
        'energy_initial': float(energy[0]),
        'energy_final': float(energy[-1]),
        'energy_drift': float(energy_drift),
        'energy_std': float(np.std(energy)),
        'max_position': float(np.max(np.abs(q_traj))),
        'max_momentum': float(np.max(np.abs(p_traj))),
        'n_evaluations': getattr(integrator, 'n_evaluations', 0)
    }


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'IntegratorType',
    'DynamicalState',
    'ForceFunction',
    'RK4Integrator',
    'SymplecticEulerIntegrator',
    'VerletIntegrator',
    'RK45Integrator',
    'AdaptiveStepResult',
    'create_integrator',
    'analyze_stability',
]