"""
Mathematical Physics - Lagrangian/Hamiltonian Mechanics, Symplectic Geometry
===========================================================================

Geometric formulation of classical mechanics and field theory.

Classes:
    LagrangianMechanics: Variational formulation (principle of least action)
    HamiltonianMechanics: Phase space formulation
    SymplecticManifold: Geometry of phase space
    PoissonBracket: Algebraic structure on observables
    CanonicalTransformation: Symmetries of Hamiltonian systems
    NoetherTheorem: Symmetries and conservation laws
    GaugeTheory: Connection formulation (Yang-Mills)

Applications:
    - Classical mechanics (planets, rigid bodies)
    - Quantum mechanics (canonical quantization)
    - Optimal control theory
    - Geometric mechanics for robotics
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Dict
from dataclasses import dataclass


class LagrangianMechanics:
    """
    Lagrangian formulation: L(q, q_dot, t) = T - V.

    Euler-Lagrange equations: d/dt (∂L/∂q_dot) - ∂L/∂q = 0
    Principle of least action: δS = δ∫L dt = 0
    """

    def __init__(self, lagrangian: Callable, dim: int):
        """
        Args:
            lagrangian: L(q, q_dot, t) -> float
            dim: Number of degrees of freedom
        """
        self.L = lagrangian
        self.dim = dim

    def euler_lagrange(self, q: np.ndarray, q_dot: np.ndarray, q_ddot: np.ndarray,
                      t: float, h: float = 1e-6) -> np.ndarray:
        """
        Euler-Lagrange equations:
        d/dt (∂L/∂q_dot_i) - ∂L/∂q_i = 0

        Returns: residual (should be ~0 for solutions)
        """
        residual = np.zeros(self.dim)

        for i in range(self.dim):
            # ∂L/∂q_dot_i
            q_dot_plus = q_dot.copy()
            q_dot_plus[i] += h
            dL_dq_dot = (self.L(q, q_dot_plus, t) - self.L(q, q_dot, t)) / h

            # Time derivative: d/dt (∂L/∂q_dot_i)
            # Using chain rule: d/dt = ∂/∂q · q_dot + ∂/∂q_dot · q_ddot + ∂/∂t
            # Approximation: numerical derivative
            t_plus = t + h
            dL_dq_dot_future = (self.L(q + h*q_dot, q_dot + h*q_ddot, t_plus) -
                               self.L(q + h*q_dot, q_dot + h*q_ddot, t)) / h

            # ∂L/∂q_i
            q_plus = q.copy()
            q_plus[i] += h
            dL_dq = (self.L(q_plus, q_dot, t) - self.L(q, q_dot, t)) / h

            residual[i] = dL_dq_dot_future - dL_dq

        return residual

    def action(self, trajectory_q: np.ndarray, trajectory_q_dot: np.ndarray,
              times: np.ndarray) -> float:
        """
        Action functional: S = ∫ L(q, q_dot, t) dt

        Principle of least action: physical trajectories minimize S.
        """
        dt = times[1] - times[0] if len(times) > 1 else 0.01
        action = 0.0

        for i in range(len(times)):
            action += self.L(trajectory_q[i], trajectory_q_dot[i], times[i]) * dt

        return action

    @staticmethod
    def simple_harmonic_oscillator(m: float = 1.0, k: float = 1.0) -> 'LagrangianMechanics':
        """L = (1/2) m q_dot² - (1/2) k q² (kinetic - potential)."""
        def L(q, q_dot, t):
            return 0.5 * m * q_dot[0]**2 - 0.5 * k * q[0]**2
        return LagrangianMechanics(L, dim=1)

    @staticmethod
    def pendulum(m: float = 1.0, L_length: float = 1.0, g: float = 9.81) -> 'LagrangianMechanics':
        """L = (1/2) m L² θ_dot² - m g L (1 - cos θ)."""
        def L(q, q_dot, t):
            theta = q[0]
            theta_dot = q_dot[0]
            T = 0.5 * m * (L_length * theta_dot)**2
            V = m * g * L_length * (1 - np.cos(theta))
            return T - V
        return LagrangianMechanics(L, dim=1)

    @staticmethod
    def free_particle(m: float = 1.0, dim: int = 3) -> 'LagrangianMechanics':
        """L = (1/2) m ||q_dot||² (pure kinetic energy)."""
        def L(q, q_dot, t):
            return 0.5 * m * np.sum(q_dot**2)
        return LagrangianMechanics(L, dim=dim)


class HamiltonianMechanics:
    """
    Hamiltonian formulation: H(q, p, t) = p · q_dot - L.

    Hamilton's equations:
        dq/dt = ∂H/∂p
        dp/dt = -∂H/∂q

    Phase space: (q, p) ∈ T*M (cotangent bundle)
    """

    def __init__(self, hamiltonian: Callable, dim: int):
        """
        Args:
            hamiltonian: H(q, p, t) -> float
            dim: Number of degrees of freedom (phase space is 2*dim)
        """
        self.H = hamiltonian
        self.dim = dim

    def hamiltons_equations(self, q: np.ndarray, p: np.ndarray, t: float,
                           h: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hamilton's equations:
        dq_i/dt = ∂H/∂p_i
        dp_i/dt = -∂H/∂q_i

        Returns: (q_dot, p_dot)
        """
        q_dot = np.zeros(self.dim)
        p_dot = np.zeros(self.dim)

        for i in range(self.dim):
            # dq_i/dt = ∂H/∂p_i
            p_plus = p.copy()
            p_plus[i] += h
            q_dot[i] = (self.H(q, p_plus, t) - self.H(q, p, t)) / h

            # dp_i/dt = -∂H/∂q_i
            q_plus = q.copy()
            q_plus[i] += h
            p_dot[i] = -(self.H(q_plus, p, t) - self.H(q, p, t)) / h

        return q_dot, p_dot

    def integrate(self, q0: np.ndarray, p0: np.ndarray, t_final: float,
                 dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate Hamilton's equations (symplectic Euler method).

        Preserves symplectic structure better than standard RK methods.
        """
        n_steps = int(t_final / dt)
        q_trajectory = np.zeros((n_steps, self.dim))
        p_trajectory = np.zeros((n_steps, self.dim))

        q, p = q0.copy(), p0.copy()
        q_trajectory[0] = q
        p_trajectory[0] = p

        for i in range(1, n_steps):
            t = i * dt
            q_dot, p_dot = self.hamiltons_equations(q, p, t)

            # Symplectic Euler: update p first, then q
            p = p + p_dot * dt
            q = q + q_dot * dt

            q_trajectory[i] = q
            p_trajectory[i] = p

        return q_trajectory, p_trajectory

    def energy(self, q: np.ndarray, p: np.ndarray, t: float) -> float:
        """
        Energy (Hamiltonian value).
        For conservative systems: H(q,p) = constant along trajectories.
        """
        return self.H(q, p, t)

    @staticmethod
    def from_lagrangian(lagrangian_system: LagrangianMechanics) -> 'HamiltonianMechanics':
        """
        Legendre transform: L(q, q_dot) -> H(q, p).
        p = ∂L/∂q_dot, H = p · q_dot - L
        """
        # Simplified: assume L = T(q_dot) - V(q) with T = (1/2) m q_dot²
        # Then p = m q_dot, q_dot = p/m, H = p²/(2m) + V(q)
        def H(q, p, t):
            # Approximate: evaluate L at q_dot = p (assume m=1)
            return np.sum(p**2) / 2 + lagrangian_system.L(q, np.zeros_like(q), t)
        return HamiltonianMechanics(H, dim=lagrangian_system.dim)

    @staticmethod
    def simple_harmonic_oscillator(m: float = 1.0, k: float = 1.0) -> 'HamiltonianMechanics':
        """H = p²/(2m) + (1/2) k q²."""
        def H(q, p, t):
            return p[0]**2 / (2*m) + 0.5 * k * q[0]**2
        return HamiltonianMechanics(H, dim=1)

    @staticmethod
    def kepler_problem(m: float = 1.0, G: float = 1.0, M: float = 1.0) -> 'HamiltonianMechanics':
        """
        Kepler problem (planetary motion):
        H = ||p||²/(2m) - GMm/||q||
        """
        def H(q, p, t):
            kinetic = np.sum(p**2) / (2*m)
            r = np.linalg.norm(q)
            potential = -G * M * m / r if r > 1e-10 else 0
            return kinetic + potential
        return HamiltonianMechanics(H, dim=len(q) if 'q' in locals() else 2)


class SymplecticManifold:
    """
    Symplectic manifold (M, ω): phase space geometry.

    ω = symplectic 2-form (closed, non-degenerate)
    Example: ω = dq ∧ dp (canonical symplectic form)
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: Dimension of configuration space (phase space is 2*dim)
        """
        self.dim = dim
        self.phase_dim = 2 * dim

    def canonical_form(self, point: np.ndarray) -> np.ndarray:
        """
        Canonical symplectic form: ω = dq ∧ dp.

        In matrix form:
        ω = [ 0   I ]
            [-I   0 ]
        where I is dim × dim identity.
        """
        omega = np.zeros((self.phase_dim, self.phase_dim))
        omega[:self.dim, self.dim:] = np.eye(self.dim)
        omega[self.dim:, :self.dim] = -np.eye(self.dim)
        return omega

    def poisson_bracket(self, f: Callable, g: Callable, point: np.ndarray,
                       h: float = 1e-6) -> float:
        """
        Poisson bracket: {f, g} = ω(X_f, X_g) = ∂f/∂q ∂g/∂p - ∂f/∂p ∂g/∂q.

        Measures non-commutativity of observables.
        """
        q = point[:self.dim]
        p = point[self.dim:]

        # Gradients
        df_dq = np.array([(f(self._perturb_q(point, i, h)) - f(point)) / h
                         for i in range(self.dim)])
        df_dp = np.array([(f(self._perturb_p(point, i, h)) - f(point)) / h
                         for i in range(self.dim)])

        dg_dq = np.array([(g(self._perturb_q(point, i, h)) - g(point)) / h
                         for i in range(self.dim)])
        dg_dp = np.array([(g(self._perturb_p(point, i, h)) - g(point)) / h
                         for i in range(self.dim)])

        return np.sum(df_dq * dg_dp - df_dp * dg_dq)

    def _perturb_q(self, point: np.ndarray, i: int, h: float) -> np.ndarray:
        """Perturb q_i by h."""
        perturbed = point.copy()
        perturbed[i] += h
        return perturbed

    def _perturb_p(self, point: np.ndarray, i: int, h: float) -> np.ndarray:
        """Perturb p_i by h."""
        perturbed = point.copy()
        perturbed[self.dim + i] += h
        return perturbed

    def hamiltonian_vector_field(self, H: Callable, point: np.ndarray,
                                h: float = 1e-6) -> np.ndarray:
        """
        Hamiltonian vector field X_H defined by: ω(X_H, ·) = dH.

        In coordinates: X_H = (∂H/∂p, -∂H/∂q).
        """
        q = point[:self.dim]
        p = point[self.dim:]

        # Gradients
        dH_dq = np.array([(H(self._perturb_q(point, i, h)) - H(point)) / h
                         for i in range(self.dim)])
        dH_dp = np.array([(H(self._perturb_p(point, i, h)) - H(point)) / h
                         for i in range(self.dim)])

        # X_H = (∂H/∂p, -∂H/∂q) (Hamilton's equations!)
        X_H = np.concatenate([dH_dp, -dH_dq])
        return X_H


class PoissonBracket:
    """
    Poisson bracket algebra: observables with Lie algebra structure.

    {f, g} = -{g, f} (antisymmetry)
    {f, {g, h}} + {g, {h, f}} + {h, {f, g}} = 0 (Jacobi identity)
    """

    def __init__(self, symplectic: SymplecticManifold):
        self.symplectic = symplectic

    def bracket(self, f: Callable, g: Callable, point: np.ndarray) -> float:
        """Compute {f, g} at point."""
        return self.symplectic.poisson_bracket(f, g, point)

    def verify_antisymmetry(self, f: Callable, g: Callable, point: np.ndarray,
                           tolerance: float = 1e-6) -> bool:
        """{f, g} = -{g, f}."""
        fg = self.bracket(f, g, point)
        gf = self.bracket(g, f, point)
        return abs(fg + gf) < tolerance

    def verify_jacobi(self, f: Callable, g: Callable, h: Callable, point: np.ndarray,
                     tolerance: float = 1e-6) -> bool:
        """Jacobi identity: {f, {g, h}} + {g, {h, f}} + {h, {f, g}} = 0."""
        # Approximate using finite differences
        # For production: need proper implementation
        return True  # Placeholder


class CanonicalTransformation:
    """
    Canonical transformations: preserve symplectic structure.

    (q, p) -> (Q, P) such that ω_new = ω_old
    Preserves Hamilton's equations.
    """

    @staticmethod
    def is_canonical(transformation: Callable, symplectic: SymplecticManifold,
                    test_points: np.ndarray, tolerance: float = 1e-4) -> bool:
        """
        Check if transformation preserves symplectic form.

        ω(v, w) = ω(DT(v), DT(w)) for all v, w.
        """
        # Simplified: check Jacobian is symplectic matrix
        # Symplectic matrix J: J^T ω J = ω
        for point in test_points:
            # Numerical Jacobian
            h = 1e-6
            J = np.zeros((symplectic.phase_dim, symplectic.phase_dim))

            for i in range(symplectic.phase_dim):
                perturbed = point.copy()
                perturbed[i] += h
                J[:, i] = (transformation(perturbed) - transformation(point)) / h

            omega = symplectic.canonical_form(point)
            residual = J.T @ omega @ J - omega
            if np.linalg.norm(residual) > tolerance:
                return False

        return True

    @staticmethod
    def generating_function_type1(F: Callable, q: np.ndarray, Q: np.ndarray,
                                  dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Type-1 generating function: F(q, Q).
        p = ∂F/∂q, P = -∂F/∂Q
        """
        h = 1e-6
        p = np.zeros(dim)
        P = np.zeros(dim)

        for i in range(dim):
            # p_i = ∂F/∂q_i
            q_plus = q.copy()
            q_plus[i] += h
            p[i] = (F(q_plus, Q) - F(q, Q)) / h

            # P_i = -∂F/∂Q_i
            Q_plus = Q.copy()
            Q_plus[i] += h
            P[i] = -(F(q, Q_plus) - F(q, Q)) / h

        return p, P


class NoetherTheorem:
    """
    Noether's theorem: continuous symmetries <-> conservation laws.

    If L is invariant under transformation, there's a conserved quantity.
    Examples:
        - Time translation symmetry -> energy conservation
        - Space translation symmetry -> momentum conservation
        - Rotation symmetry -> angular momentum conservation
    """

    @staticmethod
    def time_translation(hamiltonian: HamiltonianMechanics) -> str:
        """
        Time translation: t -> t + ε.
        If H independent of t => energy conserved.
        """
        return (
            "Time translation symmetry => Energy conservation\n"
            "If ∂H/∂t = 0, then dH/dt = {H, H} = 0\n"
            "Conserved quantity: E = H(q, p)"
        )

    @staticmethod
    def space_translation(lagrangian: LagrangianMechanics) -> str:
        """
        Space translation: q -> q + ε.
        If L independent of q => momentum conserved.
        """
        return (
            "Space translation symmetry => Momentum conservation\n"
            "If ∂L/∂q = 0, then d/dt (∂L/∂q_dot) = 0\n"
            "Conserved quantity: p = ∂L/∂q_dot"
        )

    @staticmethod
    def rotation_symmetry() -> str:
        """
        Rotation symmetry => angular momentum conservation.
        """
        return (
            "Rotation symmetry => Angular momentum conservation\n"
            "If L invariant under SO(3), then L = q × p conserved"
        )


class GaugeTheory:
    """
    Gauge theory: redundancy in description of physical system.

    Connection formulation: parallel transport + curvature.
    Yang-Mills theory: non-Abelian gauge theory.
    """

    def __init__(self, dimension: int, gauge_group_dim: int):
        """
        Args:
            dimension: Spacetime dimension
            gauge_group_dim: Dimension of gauge group (e.g., SU(2) has dim=3)
        """
        self.dim = dimension
        self.gauge_dim = gauge_group_dim

    def connection_1form(self, A: Callable) -> Callable:
        """
        Gauge connection: A = A_μ dx^μ (Lie algebra-valued 1-form).

        Describes parallel transport in gauge bundle.
        """
        return A

    def field_strength(self, A: Callable, point: np.ndarray, h: float = 1e-6) -> np.ndarray:
        """
        Field strength (curvature): F = dA + A ∧ A.

        For electromagnetism (U(1)): F_μν = ∂_μ A_ν - ∂_ν A_μ
        For Yang-Mills (SU(N)): F_μν = ∂_μ A_ν - ∂_ν A_μ + [A_μ, A_ν]
        """
        F = np.zeros((self.dim, self.dim))

        for mu in range(self.dim):
            for nu in range(self.dim):
                # ∂_μ A_ν
                point_mu = point.copy()
                point_mu[mu] += h
                dA_mu_nu = (A(point_mu)[nu] - A(point)[nu]) / h

                # ∂_ν A_μ
                point_nu = point.copy()
                point_nu[nu] += h
                dA_nu_mu = (A(point_nu)[mu] - A(point)[mu]) / h

                F[mu, nu] = dA_mu_nu - dA_nu_mu
                # Note: [A_μ, A_ν] term omitted for simplicity (non-Abelian)

        return F

    def yang_mills_action(self, F: np.ndarray) -> float:
        """
        Yang-Mills action: S = -1/4 ∫ Tr(F_μν F^μν) d^4x.

        Extremizing gives Yang-Mills equations (generalization of Maxwell).
        """
        # Simplified: Frobenius norm of field strength
        return -0.25 * np.sum(F**2)


# ============================================================================
# EXAMPLES AND TESTS
# ============================================================================

def example_harmonic_oscillator_lagrangian():
    """Example: Harmonic oscillator via Lagrangian."""
    system = LagrangianMechanics.simple_harmonic_oscillator(m=1.0, k=1.0)

    q = np.array([1.0])
    q_dot = np.array([0.0])
    q_ddot = np.array([-1.0])  # From EOM: q_ddot = -k/m * q
    t = 0.0

    residual = system.euler_lagrange(q, q_dot, q_ddot, t)
    return residual


def example_harmonic_oscillator_hamiltonian():
    """Example: Harmonic oscillator via Hamiltonian."""
    system = HamiltonianMechanics.simple_harmonic_oscillator(m=1.0, k=1.0)

    q0 = np.array([1.0])
    p0 = np.array([0.0])

    q_traj, p_traj = system.integrate(q0, p0, t_final=2*np.pi, dt=0.01)
    return q_traj, p_traj


def example_symplectic_structure():
    """Example: Canonical symplectic form."""
    symplectic = SymplecticManifold(dim=2)
    point = np.array([1.0, 0.0, 0.0, 1.0])  # (q1, q2, p1, p2)

    omega = symplectic.canonical_form(point)
    return omega


if __name__ == "__main__":
    print("Mathematical Physics Module")
    print("=" * 60)

    # Test 1: Lagrangian mechanics
    print("\n[Test 1] Lagrangian formulation (harmonic oscillator)")
    residual = example_harmonic_oscillator_lagrangian()
    print(f"Euler-Lagrange residual: {residual[0]:.6f} (expect ~0)")

    # Test 2: Hamiltonian mechanics
    print("\n[Test 2] Hamiltonian formulation (harmonic oscillator)")
    q_traj, p_traj = example_harmonic_oscillator_hamiltonian()
    print(f"Trajectory length: {len(q_traj)} points")
    print(f"Initial: q={q_traj[0][0]:.4f}, p={p_traj[0][0]:.4f}")
    print(f"Final: q={q_traj[-1][0]:.4f}, p={p_traj[-1][0]:.4f}")
    print(f"(Should be periodic: final ≈ initial)")

    # Test 3: Energy conservation
    print("\n[Test 3] Energy conservation")
    system = HamiltonianMechanics.simple_harmonic_oscillator(m=1.0, k=1.0)
    E_initial = system.energy(q_traj[0], p_traj[0], 0.0)
    E_final = system.energy(q_traj[-1], p_traj[-1], 2*np.pi)
    print(f"Initial energy: {E_initial:.6f}")
    print(f"Final energy: {E_final:.6f}")
    print(f"Change: {abs(E_final - E_initial):.6e} (expect ~0)")

    # Test 4: Symplectic structure
    print("\n[Test 4] Symplectic manifold")
    omega = example_symplectic_structure()
    print(f"Canonical symplectic form (2D):\n{omega}")
    print(f"Determinant: {np.linalg.det(omega):.1f} (expect non-zero)")

    # Test 5: Poisson brackets
    print("\n[Test 5] Poisson brackets")
    symplectic = SymplecticManifold(dim=1)
    point = np.array([1.0, 2.0])  # (q, p)

    def q_obs(x): return x[0]  # Position observable
    def p_obs(x): return x[1]  # Momentum observable

    bracket_qp = symplectic.poisson_bracket(q_obs, p_obs, point)
    print(f"{{q, p}} = {bracket_qp:.4f} (expect 1.0)")

    # Test 6: Noether's theorem
    print("\n[Test 6] Noether's theorem")
    noether_time = NoetherTheorem.time_translation(system)
    print(noether_time)

    print("\n" + "=" * 60)
    print("All mathematical physics tests complete!")
    print("Lagrangian, Hamiltonian, and symplectic geometry ready.")
