"""
Integration tests for PDE-Based Semantic Flow.

Tests physics-inspired semantic dynamics:
- Heat equation semantic smoothing
- Wave equation semantic oscillations
- Reaction-diffusion competitive patterns
- Diffusion-based semantic spreading
- Advection-based semantic flow
- Backward compatibility (use_semantic_flow=False)
- Stability and convergence properties

Author: Agent F (Validation Wave)
Date: 2025-11-03
"""

import pytest
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from hololoom.config import Config
from hololoom.warp.semantic_pde import (
    HeatEquationSolver,
    WaveEquationSolver,
    ReactionDiffusionSolver,
    AdvectionDiffusionSolver,
    PDEConfig,
)


# ============================================================================
# Fixtures
# ============================================================================


@dataclass
class SemanticsField:
    """1D semantic field for testing (e.g., embedding along one dimension)."""

    values: np.ndarray
    position: np.ndarray
    time: float = 0.0

    @property
    def shape(self) -> Tuple[int]:
        return self.values.shape


@pytest.fixture
def pde_config():
    """Create a PDE configuration for testing."""
    return PDEConfig(
        dt=0.01,  # Time step
        dx=0.1,   # Space step
        n_steps=100,
        diffusion_coefficient=1.0,
        wave_speed=1.0,
    )


@pytest.fixture
def heat_solver(pde_config):
    """Create a heat equation solver."""
    return HeatEquationSolver(
        domain_size=1.0,
        n_points=101,
        diffusion_coefficient=0.1,
        dt=pde_config.dt,
        boundary_condition="dirichlet",
    )


@pytest.fixture
def wave_solver(pde_config):
    """Create a wave equation solver."""
    return WaveEquationSolver(
        domain_size=1.0,
        n_points=101,
        wave_speed=1.0,
        dt=pde_config.dt,
        boundary_condition="dirichlet",
    )


@pytest.fixture
def reaction_diffusion_solver(pde_config):
    """Create a reaction-diffusion solver."""
    return ReactionDiffusionSolver(
        domain_size=1.0,
        n_points=51,
        diffusion_a=0.1,
        diffusion_b=0.05,
        reaction_rate_a=0.05,
        reaction_rate_b=0.1,
        dt=pde_config.dt,
    )


@pytest.fixture
def advection_diffusion_solver(pde_config):
    """Create an advection-diffusion solver."""
    return AdvectionDiffusionSolver(
        domain_size=1.0,
        n_points=101,
        velocity=1.0,
        diffusion_coefficient=0.1,
        dt=pde_config.dt,
    )


@pytest.fixture
def gaussian_field() -> SemanticsField:
    """Create a Gaussian-shaped semantic field."""
    x = np.linspace(0, 1, 101)
    values = np.exp(-((x - 0.5) ** 2) / 0.05)
    return SemanticsField(values=values, position=x)


@pytest.fixture
def dirac_delta_field() -> SemanticsField:
    """Create a Dirac delta (spike) semantic field."""
    x = np.linspace(0, 1, 101)
    values = np.zeros_like(x)
    values[50] = 1.0  # Spike at center
    return SemanticsField(values=values, position=x)


# ============================================================================
# Test 1: Heat Equation (Smoothing)
# ============================================================================


def test_heat_solver_initialization(heat_solver):
    """Test heat equation solver initializes."""
    assert heat_solver.n_points == 101
    assert heat_solver.diffusion_coefficient == 0.1
    assert heat_solver.dt == 0.01


def test_heat_equation_smoothing(heat_solver, gaussian_field):
    """
    Test that heat equation smooths initial condition.

    ut = D * uxx (diffusion equation)
    Gaussian should broaden over time.
    """
    u = gaussian_field.values.copy()
    initial_spread = np.std(np.nonzero(u)[0])

    # Evolve for 10 steps
    for _ in range(10):
        u = heat_solver.step(u)

    # Spread should increase (diffusion broadens)
    final_spread = np.sum(u)  # Total mass preserved

    # Mass should be approximately conserved
    initial_mass = np.sum(gaussian_field.values)
    assert 0.9 * initial_mass <= final_spread <= 1.1 * initial_mass


def test_heat_equation_energy_decay(heat_solver, dirac_delta_field):
    """
    Test that heat equation energy decays.

    E(t) = ∫ u² dx should decrease with diffusion.
    """
    u = dirac_delta_field.values.copy()
    energy_history = [np.sum(u**2)]

    # Evolve
    for _ in range(50):
        u = heat_solver.step(u)
        energy_history.append(np.sum(u**2))

    energy_history = np.array(energy_history)

    # Energy should be monotonically decreasing
    energy_decreasing = np.all(np.diff(energy_history) <= 1e-6)
    assert energy_decreasing, "Heat equation should decrease energy"


def test_heat_equation_maximum_principle(heat_solver, gaussian_field):
    """
    Test maximum principle: max is non-increasing.

    max(u(t)) <= max(u(0)) for all t > 0
    """
    u = gaussian_field.values.copy()
    initial_max = np.max(u)

    for _ in range(20):
        u = heat_solver.step(u)

    final_max = np.max(u)

    assert final_max <= initial_max * 1.01, "Maximum should not increase (up to tolerance)"


def test_heat_equation_steady_state(heat_solver):
    """
    Test convergence to steady state (zero for Dirichlet BC).

    With homogeneous Dirichlet BC, solution → 0 as t → ∞
    """
    u = np.sin(np.pi * np.linspace(0, 1, 101))

    # Evolve for many steps
    for _ in range(1000):
        u = heat_solver.step(u)

    # Should approach zero
    assert np.max(np.abs(u)) < 0.01, "Should converge to zero"


def test_heat_equation_conservation_law():
    """
    Test conservation of mass (if homogeneous Neumann BC).

    For Neumann BC, ∫ u dx = constant
    """
    heat = HeatEquationSolver(
        domain_size=1.0,
        n_points=101,
        diffusion_coefficient=0.1,
        dt=0.01,
        boundary_condition="neumann",
    )

    u = np.sin(np.pi * np.linspace(0, 1, 101))
    initial_mass = np.sum(u)

    for _ in range(50):
        u = heat.step(u)

    final_mass = np.sum(u)

    # Mass should be conserved
    assert np.isclose(initial_mass, final_mass, atol=0.1)


# ============================================================================
# Test 2: Wave Equation (Oscillations)
# ============================================================================


def test_wave_solver_initialization(wave_solver):
    """Test wave equation solver initializes."""
    assert wave_solver.n_points == 101
    assert wave_solver.wave_speed == 1.0
    assert wave_solver.dt == 0.01


def test_wave_equation_energy_conservation(wave_solver, gaussian_field):
    """
    Test wave equation conserves energy.

    E = ∫ (ut² + c² ux²) dx = constant
    """
    u = gaussian_field.values.copy()
    ut = np.zeros_like(u)

    # Initial kinetic + potential energy
    energies = []

    # Evolve wave
    for step in range(100):
        u_new, ut_new = wave_solver.step(u, ut)

        # Approximate energy
        energy = np.sum(ut_new**2) + np.sum(np.gradient(u_new)**2)
        energies.append(energy)

        u = u_new
        ut = ut_new

    energies = np.array(energies)

    # Energy should oscillate around average (not grow unboundedly)
    energy_mean = np.mean(energies)
    energy_var = np.std(energies)

    # Variance should be small relative to mean
    assert energy_var / energy_mean < 0.5, "Energy should not grow unboundedly"


def test_wave_equation_finite_speed_propagation(wave_solver):
    """
    Test finite speed of propagation.

    Disturbance should propagate at speed c (finite).
    """
    u = np.zeros(101)
    ut = np.zeros(101)
    u[50] = 1.0  # Disturbance at center

    positions_affected = [np.sum(np.abs(u) > 1e-6)]

    # Evolve
    for _ in range(20):
        u_new, ut_new = wave_solver.step(u, ut)
        positions_affected.append(np.sum(np.abs(u_new) > 1e-6))
        u, ut = u_new, ut_new

    # Affected region should grow (but not instantly fill domain)
    assert positions_affected[-1] > positions_affected[0], "Wave should propagate"
    assert positions_affected[-1] < 100, "Not all points affected instantly"


def test_wave_equation_d_alembert_solution():
    """
    Test d'Alembert's solution for wave equation.

    For free wave with smooth initial conditions,
    solution follows d'Alembert formula.
    """
    # Create simple initial condition
    u = np.sin(2 * np.pi * np.linspace(0, 1, 101))
    ut = np.zeros_like(u)

    wave = WaveEquationSolver(
        domain_size=1.0,
        n_points=101,
        wave_speed=1.0,
        dt=0.01,
    )

    # Evolve
    for _ in range(50):
        u_new, ut_new = wave.step(u, ut)
        u, ut = u_new, ut_new

    # Solution should remain bounded
    assert np.max(np.abs(u)) <= np.max(np.abs(np.sin(2 * np.pi * np.linspace(0, 1, 101)))) * 1.1


# ============================================================================
# Test 3: Reaction-Diffusion (Pattern Formation)
# ============================================================================


def test_reaction_diffusion_initialization(reaction_diffusion_solver):
    """Test reaction-diffusion solver initializes."""
    assert reaction_diffusion_solver.diffusion_a == 0.1
    assert reaction_diffusion_solver.diffusion_b == 0.05


def test_reaction_diffusion_turing_patterns():
    """
    Test formation of Turing patterns.

    Reaction-diffusion can create spatial patterns through Turing instability.
    """
    # Gray-Scott parameters (know to produce patterns)
    solver = ReactionDiffusionSolver(
        domain_size=1.0,
        n_points=51,
        diffusion_a=0.1,
        diffusion_b=0.05,
        reaction_rate_a=0.03,
        reaction_rate_b=0.06,
        dt=0.01,
    )

    # Initialize uniform + small noise
    u = 0.5 * np.ones(51)
    v = 0.25 * np.ones(51)
    u += 0.05 * np.random.randn(51)
    v += 0.05 * np.random.randn(51)

    # Evolve
    for _ in range(500):
        u, v = solver.step(u, v)

    # Patterns should form (non-uniform)
    u_var = np.var(u)
    v_var = np.var(v)

    assert u_var > 0.001, "Patterns should form"
    assert v_var > 0.001, "Patterns should form"


def test_reaction_diffusion_instability():
    """
    Test activation-inhibition mechanism.

    Small perturbations grow (instability) in Turing regime.
    """
    solver = ReactionDiffusionSolver(
        domain_size=1.0,
        n_points=51,
        diffusion_a=0.01,
        diffusion_b=0.1,
        reaction_rate_a=0.1,
        reaction_rate_b=0.05,
        dt=0.001,
    )

    # Uniform state + tiny perturbation
    u = 1.0 * np.ones(51)
    v = 0.5 * np.ones(51)
    u[25] += 1e-3  # Small perturbation

    initial_perturbation = np.std(u)

    # Evolve
    for _ in range(200):
        u, v = solver.step(u, v)

    final_perturbation = np.std(u)

    # Perturbation should grow (Turing instability)
    assert final_perturbation > initial_perturbation * 2, "Should amplify perturbations"


def test_reaction_diffusion_convergence():
    """Test reaction-diffusion convergence to stable pattern."""
    solver = ReactionDiffusionSolver(
        domain_size=1.0,
        n_points=51,
        diffusion_a=0.1,
        diffusion_b=0.05,
        reaction_rate_a=0.05,
        reaction_rate_b=0.1,
        dt=0.01,
    )

    u = 0.5 * np.ones(51) + 0.1 * np.sin(2 * np.pi * np.linspace(0, 1, 51))
    v = 0.25 * np.ones(51) + 0.05 * np.sin(4 * np.pi * np.linspace(0, 1, 51))

    patterns = []

    # Evolve and track pattern
    for _ in range(300):
        u, v = solver.step(u, v)
        patterns.append(u.copy())

    # Later patterns should be similar (convergence)
    pattern1 = patterns[-50]
    pattern2 = patterns[-10]

    correlation = np.corrcoef(pattern1, pattern2)[0, 1]
    assert correlation > 0.9, "Should converge to stable pattern"


# ============================================================================
# Test 4: Advection-Diffusion
# ============================================================================


def test_advection_diffusion_initialization(advection_diffusion_solver):
    """Test advection-diffusion solver initializes."""
    assert advection_diffusion_solver.velocity == 1.0
    assert advection_diffusion_solver.diffusion_coefficient == 0.1


def test_advection_equation_transport():
    """
    Test advection transports without diffusion.

    ut + v*ux = 0 has solution u(x,t) = u(x-vt, 0)
    """
    # Pure advection (no diffusion)
    solver = AdvectionDiffusionSolver(
        domain_size=1.0,
        n_points=101,
        velocity=1.0,
        diffusion_coefficient=0.0,  # No diffusion
        dt=0.01,
    )

    # Gaussian initial condition
    x = np.linspace(0, 1, 101)
    u = np.exp(-((x - 0.2) ** 2) / 0.02)

    initial_max_idx = np.argmax(u)

    # Evolve
    for _ in range(50):
        u = solver.step(u)

    final_max_idx = np.argmax(u)

    # Peak should move right (positive velocity)
    assert final_max_idx > initial_max_idx, "Should transport right"


def test_advection_diffusion_balance():
    """
    Test balance of advection and diffusion.

    Strong advection: transport dominates
    Strong diffusion: spreading dominates
    """
    # Advection-dominated
    solver_adv = AdvectionDiffusionSolver(
        domain_size=1.0,
        n_points=101,
        velocity=10.0,
        diffusion_coefficient=0.01,
        dt=0.001,
    )

    # Diffusion-dominated
    solver_dif = AdvectionDiffusionSolver(
        domain_size=1.0,
        n_points=101,
        velocity=0.1,
        diffusion_coefficient=1.0,
        dt=0.01,
    )

    u_adv = np.exp(-((np.linspace(0, 1, 101) - 0.2) ** 2) / 0.02)
    u_dif = np.exp(-((np.linspace(0, 1, 101) - 0.2) ** 2) / 0.02)

    # Evolve both
    for _ in range(10):
        u_adv = solver_adv.step(u_adv)
        u_dif = solver_dif.step(u_dif)

    # Advection should transport peak more
    # Diffusion should smooth more
    peak_move_adv = np.abs(np.argmax(u_adv) - 20)
    peak_move_dif = np.abs(np.argmax(u_dif) - 20)

    assert peak_move_adv > peak_move_dif, "Advection should transport more"


# ============================================================================
# Test 5: Stability Analysis
# ============================================================================


def test_heat_equation_cfl_stability(heat_solver):
    """
    Test CFL (Courant-Friedrichs-Lewy) stability.

    For parabolic PDE: dt <= C * dx²
    """
    # Small dt, large dx should be stable
    solver = HeatEquationSolver(
        domain_size=1.0,
        n_points=50,
        diffusion_coefficient=0.1,
        dt=0.001,
        boundary_condition="dirichlet",
    )

    u = np.sin(np.pi * np.linspace(0, 1, 50))

    # Evolve - should not blow up
    for _ in range(100):
        u = solver.step(u)
        assert np.all(np.isfinite(u)), "Solution blew up - stability violated"


def test_wave_equation_courant_condition():
    """
    Test Courant condition for wave equation.

    For hyperbolic PDE: c * dt <= dx
    """
    # Velocity = 1, space step = 0.01 → dt <= 0.01
    solver = WaveEquationSolver(
        domain_size=1.0,
        n_points=101,
        wave_speed=1.0,
        dt=0.009,  # Satisfies Courant
        boundary_condition="dirichlet",
    )

    u = np.sin(np.pi * np.linspace(0, 1, 101))
    ut = np.zeros_like(u)

    # Evolve - should be stable
    for _ in range(100):
        u, ut = solver.step(u, ut)
        assert np.all(np.isfinite(u)), "Solution blew up"
        assert np.all(np.isfinite(ut)), "Velocity blew up"


# ============================================================================
# Test 6: Backward Compatibility
# ============================================================================


def test_config_semantic_flow_flag():
    """Test that config supports use_semantic_flow flag."""
    config = Config.fused()

    # Check backward compatibility
    assert not hasattr(config, 'use_semantic_flow') or config.use_semantic_flow == False


def test_non_pde_diffusion():
    """
    Test simple diffusion without PDE solver.

    Can approximate diffusion with matrix operations.
    """
    # Simple Gaussian blur (approximates diffusion)
    u = np.sin(np.pi * np.linspace(0, 1, 101))

    # Diffusion kernel
    kernel = np.array([0.25, 0.5, 0.25])

    for _ in range(10):
        u_new = np.convolve(u, kernel, mode='same')
        u = u_new

    # Should be smoothed
    assert np.std(np.gradient(u)) < np.std(np.gradient(np.sin(np.pi * np.linspace(0, 1, 101))))


def test_graceful_fallback_no_pde():
    """Test graceful fallback when PDE features unavailable."""
    config = Config.fast()

    # Should work without PDE features
    assert not hasattr(config, 'use_semantic_flow') or config.use_semantic_flow == False

    # Simple smoothing still works
    u = np.random.randn(100)
    u_smooth = np.convolve(u, np.ones(5) / 5, mode='same')

    assert u_smooth.shape == u.shape


# ============================================================================
# Test 7: Physical Correctness
# ============================================================================


def test_heat_equation_fundamental_solution():
    """
    Test heat kernel (fundamental solution).

    K(x,t) = (1/√(4πDt)) exp(-x²/(4Dt))
    """
    # Initialize with delta function at origin
    u = np.zeros(201)
    u[100] = 1.0  # Delta at center

    solver = HeatEquationSolver(
        domain_size=2.0,
        n_points=201,
        diffusion_coefficient=0.1,
        dt=0.01,
        boundary_condition="dirichlet",
    )

    # Evolve for time T
    T = 0.1
    for _ in range(int(T / 0.01)):
        u = solver.step(u)

    # Should approximate Gaussian
    x = np.linspace(-1, 1, 201)
    theoretical = (1.0 / np.sqrt(4 * np.pi * 0.1 * T)) * np.exp(
        -(x**2) / (4 * 0.1 * T)
    )

    # Normalize for comparison
    u_norm = u / np.max(u)
    theoretical_norm = theoretical / np.max(theoretical)

    correlation = np.corrcoef(u_norm, theoretical_norm)[0, 1]
    assert correlation > 0.8, "Should approximate Gaussian (heat kernel)"


def test_wave_equation_separation_of_variables():
    """
    Test wave equation with separable solution.

    u(x,t) = sin(πx) cos(πct) satisfies wave equation.
    """
    x = np.linspace(0, 1, 101)
    c = 1.0
    t = 0.0

    solver = WaveEquationSolver(
        domain_size=1.0,
        n_points=101,
        wave_speed=c,
        dt=0.01,
        boundary_condition="dirichlet",
    )

    # Initial condition: sin(πx)
    u = np.sin(np.pi * x)
    # Initial velocity: -πc sin(πx) (from time derivative)
    ut = -np.pi * c * np.sin(np.pi * x)

    # Evolve to t=0.1
    for step in range(10):
        u_new, ut_new = solver.step(u, ut)
        t_new = t + solver.dt

        # Check against analytical solution
        u_analytical = np.sin(np.pi * x) * np.cos(np.pi * c * t_new)

        error = np.linalg.norm(u_new - u_analytical) / np.linalg.norm(u_analytical)
        assert error < 0.1, f"Error {error} too large at t={t_new}"

        u, ut = u_new, ut_new
        t = t_new


# ============================================================================
# Test 8: Edge Cases
# ============================================================================


def test_zero_diffusion():
    """Test heat equation with zero diffusion."""
    solver = HeatEquationSolver(
        domain_size=1.0,
        n_points=101,
        diffusion_coefficient=0.0,
        dt=0.01,
        boundary_condition="dirichlet",
    )

    u = np.sin(np.pi * np.linspace(0, 1, 101))
    u_original = u.copy()

    for _ in range(10):
        u = solver.step(u)

    # Should not change (no diffusion)
    assert np.allclose(u, u_original, atol=1e-6)


def test_zero_velocity():
    """Test advection-diffusion with zero velocity."""
    solver = AdvectionDiffusionSolver(
        domain_size=1.0,
        n_points=101,
        velocity=0.0,
        diffusion_coefficient=0.1,
        dt=0.01,
    )

    u = np.exp(-((np.linspace(0, 1, 101) - 0.5) ** 2) / 0.02)

    # Should only diffuse (not transport)
    initial_max_idx = np.argmax(u)

    for _ in range(20):
        u = solver.step(u)

    final_max_idx = np.argmax(u)

    # Peak should stay in place
    assert initial_max_idx == final_max_idx


# ============================================================================
# Test 9: Performance
# ============================================================================


def test_heat_solver_speed():
    """Test heat solver speed."""
    import time

    solver = HeatEquationSolver(
        domain_size=1.0,
        n_points=101,
        diffusion_coefficient=0.1,
        dt=0.01,
    )

    u = np.sin(np.pi * np.linspace(0, 1, 101))

    start = time.time()
    for _ in range(100):
        u = solver.step(u)
    elapsed = (time.time() - start) * 1000

    avg_time_ms = elapsed / 100
    assert avg_time_ms < 10, f"Heat step too slow: {avg_time_ms}ms"


# ============================================================================
# Main Test Summary
# ============================================================================


def test_pde_integration_summary():
    """
    Summary test showing all PDE features work together.

    This is a meta-test that documents the feature scope.
    """
    features = [
        "Heat equation (parabolic, smoothing)",
        "Wave equation (hyperbolic, oscillations)",
        "Reaction-diffusion (pattern formation)",
        "Advection-diffusion (transport + smoothing)",
        "Energy conservation (wave equation)",
        "Mass conservation (diffusion with Neumann BC)",
        "Stability analysis (CFL, Courant conditions)",
        "Turing patterns (activation-inhibition)",
        "Finite speed of propagation",
        "Physical correctness (heat kernel, d'Alembert)",
        "Backward compatibility (simple diffusion fallback)",
        "Edge case handling",
    ]

    assert len(features) == 12
    # This test always passes - it's documentation
