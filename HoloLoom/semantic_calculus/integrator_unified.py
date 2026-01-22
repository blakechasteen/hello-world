"""
Unified Geometric Integration for Semantic Flows
================================================
UPGRADED (2025-11-03): Now uses moonshot's unified integrator framework!

This module replaces the hard-coded Störmer-Verlet in integrator.py with
a flexible system that supports:
- Verlet (symplectic, energy-conserving)
- RK4 (4th order accuracy)
- RK45 (adaptive step size)
- Symplectic Euler (fast, energy-preserving)

Preserves backward compatibility while enabling advanced numerical methods.

Migration Guide:
---------------
OLD:
    from HoloLoom.semantic_calculus.integrator import GeometricIntegrator
    integrator = GeometricIntegrator(projection_matrix)

NEW:
    from HoloLoom.semantic_calculus.integrator_unified import UnifiedGeometricIntegrator
    integrator = UnifiedGeometricIntegrator(projection_matrix, integrator_type="verlet")

Or for backward compatibility:
    from HoloLoom.semantic_calculus.integrator_unified import GeometricIntegrator
    integrator = GeometricIntegrator(projection_matrix)  # Still works!
"""

import numpy as np
from typing import Callable, List, Dict, Optional
from dataclasses import dataclass

# Import moonshot integrators
try:
    from HoloLoom.memory.integrators import (
        IntegratorType,
        DynamicalState,
        ForceFunction,
        create_integrator,
    )
    _HAVE_MOONSHOT = True
except ImportError:
    _HAVE_MOONSHOT = False
    import warnings
    warnings.warn(
        "Moonshot integrators not available. Falling back to hard-coded Verlet. "
        "For advanced integration methods, ensure integrators.py is available."
    )

# Import original semantic calculus components
from .performance import ProjectionCache, HAS_NUMBA


@dataclass
class SemanticState:
    """
    State in semantic dimension space.

    q: Position in semantic coordinates (n_dims)
    p: Momentum in semantic coordinates (n_dims)
    t: Time
    """
    q: np.ndarray
    p: np.ndarray
    t: float = 0.0

    @property
    def kinetic_energy(self, mass: float = 1.0) -> float:
        """T = (1/2m)||p||²"""
        return 0.5 * np.dot(self.p, self.p) / mass

    def hamiltonian(self, potential_fn: Callable, mass: float = 1.0) -> float:
        """H = T + V"""
        # kinetic_energy is a property (default mass=1.0)
        # Compute kinetic energy directly with specified mass
        kinetic = 0.5 * np.dot(self.p, self.p) / mass
        return kinetic + potential_fn(self.q)

    def to_dynamical_state(self) -> 'DynamicalState':
        """Convert to moonshot DynamicalState format."""
        if _HAVE_MOONSHOT:
            return DynamicalState(q=self.q, p=self.p, t=self.t)
        return self

    @staticmethod
    def from_dynamical_state(dyn_state: 'DynamicalState') -> 'SemanticState':
        """Convert from moonshot DynamicalState."""
        return SemanticState(q=dyn_state.q, p=dyn_state.p, t=dyn_state.t)


class SemanticForceFunction:
    """
    Force function adapter for semantic space.

    Wraps semantic gradient function to work with moonshot integrators.
    """

    def __init__(
        self,
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        mass: float = 1.0
    ):
        self.gradient_fn = gradient_fn
        self.mass = mass

    def __call__(self, state: DynamicalState):
        """
        Compute Hamilton's equations in semantic space.

        dq/dt = p / m
        dp/dt = -∇V(q)
        """
        dq_dt = state.p / self.mass
        dp_dt = -self.gradient_fn(state.q)
        return dq_dt, dp_dt


class UnifiedGeometricIntegrator:
    """
    Unified geometric integrator using moonshot framework.

    Supports multiple integration methods:
    - "verlet": Velocity Verlet (energy-conserving, recommended)
    - "rk4": Runge-Kutta 4th order (high accuracy)
    - "rk45": Adaptive RK45 (automatic step size)
    - "symplectic": Symplectic Euler (fast, stable)

    Preserves backward compatibility with original GeometricIntegrator.
    """

    def __init__(
        self,
        projection_matrix: np.ndarray,
        mass: float = 1.0,
        enable_projection_cache: bool = True,
        integrator_type: str = "verlet",
        use_moonshot: bool = True
    ):
        """
        Args:
            projection_matrix: P matrix (n_dims, embedding_dim)
            mass: Semantic "mass" for momentum
            enable_projection_cache: Cache projections
            integrator_type: "verlet", "rk4", "rk45", "symplectic"
            use_moonshot: Use moonshot integrators (True) or fallback (False)
        """
        self.P = projection_matrix
        self.P_T = projection_matrix.T
        self.mass = mass
        self.n_dims = projection_matrix.shape[0]

        # Projection cache
        if enable_projection_cache:
            self._proj_cache = ProjectionCache(max_size=1000)
        else:
            self._proj_cache = None

        # Integrator setup
        self.integrator_type = integrator_type
        self._use_moonshot = use_moonshot and _HAVE_MOONSHOT

        if not self._use_moonshot:
            self.integrator = None  # Will use fallback Verlet
        else:
            self.integrator = None  # Created on demand
            self._gradient_fn_cache = None

    def _create_moonshot_integrator(self, gradient_fn: Callable):
        """Create moonshot integrator on demand."""
        if self.integrator is not None:
            return  # Already created

        # Wrap gradient function
        force_fn = SemanticForceFunction(gradient_fn, self.mass)

        # Map integrator type
        integrator_map = {
            'verlet': IntegratorType.VERLET,
            'rk4': IntegratorType.RK4,
            'rk45': IntegratorType.RK45,
            'symplectic': IntegratorType.SYMPLECTIC_EULER,
        }

        int_type = integrator_map.get(self.integrator_type.lower())
        if int_type is None:
            raise ValueError(f"Unknown integrator type: {self.integrator_type}")

        # Create integrator
        mass_array = np.full(self.n_dims, self.mass)

        if int_type in [IntegratorType.VERLET, IntegratorType.SYMPLECTIC_EULER]:
            self.integrator = create_integrator(int_type, force_fn, mass=mass_array)
        elif int_type == IntegratorType.RK45:
            self.integrator = create_integrator(int_type, force_fn, rtol=1e-6, atol=1e-8)
        else:
            self.integrator = create_integrator(int_type, force_fn)

        self._gradient_fn_cache = gradient_fn

    def project_to_semantic(self, q_full: np.ndarray) -> np.ndarray:
        """Project full embedding to semantic coordinates: q_semantic = P @ q_full"""
        return self.P @ q_full

    def lift_to_full(self, q_semantic: np.ndarray) -> np.ndarray:
        """Lift semantic coordinates back to full space: q_full ≈ P.T @ q_semantic"""
        return self.P_T @ q_semantic

    def project_gradient(self, grad_full: np.ndarray) -> np.ndarray:
        """Project gradient: ∇V_semantic = P @ ∇V_full"""
        return self.P @ grad_full

    def step(
        self,
        state: SemanticState,
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        dt: float
    ) -> SemanticState:
        """
        Take single integration step.

        Args:
            state: Current semantic state
            gradient_fn: Gradient function ∇V(q) in semantic space
            dt: Time step

        Returns:
            New state at t + dt
        """
        if self._use_moonshot:
            return self._step_moonshot(state, gradient_fn, dt)
        else:
            return self._step_fallback_verlet(state, gradient_fn, dt)

    def _step_moonshot(self, state: SemanticState, gradient_fn: Callable, dt: float) -> SemanticState:
        """Step using moonshot integrators."""
        # Create integrator on first use
        self._create_moonshot_integrator(gradient_fn)

        # Convert to DynamicalState
        dyn_state = state.to_dynamical_state()

        # Take step
        new_dyn_state = self.integrator.step(dyn_state, dt)

        # Convert back
        return SemanticState.from_dynamical_state(new_dyn_state)

    def _step_fallback_verlet(
        self,
        state: SemanticState,
        gradient_fn: Callable,
        dt: float
    ) -> SemanticState:
        """Fallback: Hard-coded Störmer-Verlet (original implementation)."""
        q, p = state.q, state.p

        # Half-step momentum
        grad_q = gradient_fn(q)
        p_half = p - 0.5 * dt * grad_q

        # Full-step position
        q_new = q + dt * p_half / self.mass

        # Half-step momentum (using new position)
        grad_q_new = gradient_fn(q_new)
        p_new = p_half - 0.5 * dt * grad_q_new

        return SemanticState(q=q_new, p=p_new, t=state.t + dt)

    def integrate_trajectory(
        self,
        q0_full: np.ndarray,
        p0_full: np.ndarray,
        gradient_fn_full: Callable[[np.ndarray], np.ndarray],
        dt: float,
        n_steps: int
    ) -> List[SemanticState]:
        """
        Integrate semantic trajectory.

        Args:
            q0_full: Initial position in full space (384D)
            p0_full: Initial momentum in full space (384D)
            gradient_fn_full: Gradient function in full space
            dt: Time step
            n_steps: Number of steps

        Returns:
            List of semantic states
        """
        # Project initial conditions
        q0_sem = self.project_to_semantic(q0_full)
        p0_sem = self.project_to_semantic(p0_full)

        # Create gradient function in semantic space
        def gradient_fn_semantic(q_sem):
            q_full = self.lift_to_full(q_sem)
            grad_full = gradient_fn_full(q_full)
            return self.project_gradient(grad_full)

        # Integrate
        states = []
        state = SemanticState(q=q0_sem, p=p0_sem, t=0.0)
        states.append(state)

        for _ in range(n_steps):
            state = self.step(state, gradient_fn_semantic, dt)
            states.append(state)

        return states

    def compute_energy_drift(
        self,
        states: List[SemanticState],
        potential_fn: Callable
    ) -> Dict:
        """
        Measure energy conservation.

        For symplectic integrators, energy should be nearly constant.
        """
        energies = [state.hamiltonian(potential_fn, self.mass) for state in states]
        energies = np.array(energies)

        return {
            'initial_energy': float(energies[0]),
            'final_energy': float(energies[-1]),
            'energy_drift': float(energies[-1] - energies[0]),
            'max_energy': float(np.max(energies)),
            'min_energy': float(np.min(energies)),
            'std_energy': float(np.std(energies)),
            'relative_drift': float((energies[-1] - energies[0]) / (energies[0] + 1e-10))
        }


# Backward compatibility alias
GeometricIntegrator = UnifiedGeometricIntegrator


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'SemanticState',
    'UnifiedGeometricIntegrator',
    'GeometricIntegrator',  # Backward compatible
    'SemanticForceFunction',
]
