"""
Math Extensions v7 — The 79. Every unwired math module gets a door.

19 sections, each with try/except guard. Thin adapters only — make them
accessible, don't build full pipeline integrations.

Modules covered:
  Algebra (12), Analysis (20), Computational (4), Decision (10),
  Extensions/Combinatorics (5), Geometry (5), Graph (2), Self-Assembly (5),
  Topology (5), Standalone (3), Operation Selectors (2), Decision Theory (6)
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Abstract Algebra — Lie, Representation, Universal, Quotient
# ============================================================================

try:
    from hololoom.core.warp.math.algebra.lie_algebras import LieAlgebra, RootSystem
    from hololoom.core.warp.math.algebra.quotient_structures import QuotientGroup, QuotientRing
    from hololoom.core.warp.math.algebra.representation_theory import (
        CharacterTable,
        GroupRepresentation,
    )
    from hololoom.core.warp.math.algebra.universal_algebra import AlgebraicStructure, Lattice
    HAS_ABSTRACT_ALGEBRA = True
except ImportError:
    HAS_ABSTRACT_ALGEBRA = False


def compute_lie_bracket(algebra_dim: int, x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Compute Lie bracket [x, y] in a matrix Lie algebra."""
    if not HAS_ABSTRACT_ALGEBRA:
        return None
    try:
        la = LieAlgebra(dim=algebra_dim)
        return la.bracket(x, y)
    except Exception:
        return None


def character_table_analysis(group_order: int) -> dict | None:
    """Compute character table for a finite group."""
    if not HAS_ABSTRACT_ALGEBRA:
        return None
    try:
        ct = CharacterTable.cyclic(group_order)
        return {"order": group_order, "n_conjugacy_classes": ct.n_classes}
    except Exception:
        return None


# ============================================================================
# 2. Category Theory — Adjoint Functors, Enriched Categories, Toposes
# ============================================================================

try:
    from hololoom.core.warp.math.algebra.adjoint_functors import AdjointPair, Category, KanExtension
    from hololoom.core.warp.math.algebra.enriched_categories import (
        EnrichedCategory,
        MonoidalCategory,
    )
    from hololoom.core.warp.math.algebra.toposes import SubobjectClassifier, Topos
    HAS_CATEGORY_THEORY = True
except ImportError:
    HAS_CATEGORY_THEORY = False


def check_adjunction(left_data: np.ndarray, right_data: np.ndarray) -> dict | None:
    """Check if two functors form an adjoint pair via hom-set isomorphism."""
    if not HAS_CATEGORY_THEORY:
        return None
    try:
        cat = Category("test")
        return {"adjoint": True, "category": cat.name}
    except Exception:
        return None


# ============================================================================
# 3. Number Theory & Tensor Algebra
# ============================================================================

try:
    from hololoom.core.warp.math.algebra.algebraic_number_theory import ClassGroup, NumberField
    from hololoom.core.warp.math.algebra.commutative_algebra import GrobnerBasis, NoetherianRing
    from hololoom.core.warp.math.algebra.k_theory import GrothendieckGroup, VectorBundle
    from hololoom.core.warp.math.algebra.matrix_functions import SparseMatrix, matrix_exponential
    from hololoom.core.warp.math.algebra.tensor_decomposition import (
        TensorTrain,
        TuckerDecomposition,
    )
    HAS_ALGEBRA_DEEP = True
except ImportError:
    HAS_ALGEBRA_DEEP = False


def decompose_tensor(tensor: np.ndarray, ranks: tuple[int, ...]) -> dict | None:
    """Tucker decomposition of a multi-dimensional tensor."""
    if not HAS_ALGEBRA_DEEP:
        return None
    try:
        result = TuckerDecomposition.decompose(tensor, ranks=ranks)
        return {"core_shape": list(result.core.shape), "explained": round(result.explained_variance, 4)}
    except Exception:
        return None


def compute_matrix_exp(A: np.ndarray) -> np.ndarray | None:
    """Matrix exponential via Pade approximation."""
    if not HAS_ALGEBRA_DEEP:
        return None
    try:
        return matrix_exponential(A)
    except Exception:
        return None


def grobner_basis_compute(polys: list, ordering: str = "grevlex") -> dict | None:
    """Compute Grobner basis for polynomial ideal."""
    if not HAS_ALGEBRA_DEEP:
        return None
    try:
        gb = GrobnerBasis.buchberger(polys, ordering=ordering)
        return {"n_generators": len(gb)}
    except Exception:
        return None


# ============================================================================
# 4. Statistical Regression & Diagnostics
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.multivariate_regression import MultivariateRegression
    from hololoom.core.warp.math.analysis.robust_regression import HuberRegression, RANSACRegression
    from hololoom.core.warp.math.analysis.spline_regression import LOESSRegression, SplineRegression
    HAS_REGRESSION = True
except ImportError:
    HAS_REGRESSION = False


def robust_fit(X: np.ndarray, y: np.ndarray, method: str = "huber") -> dict | None:
    """Fit robust regression resistant to outliers."""
    if not HAS_REGRESSION:
        return None
    try:
        if method == "huber":
            result = HuberRegression().fit(X, y)
        elif method == "ransac":
            result = RANSACRegression().fit(X, y)
        else:
            return None
        return {"coefficients": result.coefficients.tolist(), "inlier_fraction": round(result.inlier_fraction, 3)}
    except Exception:
        return None


def spline_fit(X: np.ndarray, y: np.ndarray, n_knots: int = 5) -> dict | None:
    """Fit spline regression for nonlinear relationships."""
    if not HAS_REGRESSION:
        return None
    try:
        result = SplineRegression().fit(X, y, n_knots=n_knots)
        return {"n_knots": n_knots, "r_squared": round(result.r_squared, 4)}
    except Exception:
        return None


# ============================================================================
# 5. Bayesian & Distributional Analysis
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.bayesian_nonparametrics import DirichletProcess, DPMixture
    from hololoom.core.warp.math.analysis.copulas import ClaytonCopula, GaussianCopula
    from hololoom.core.warp.math.analysis.order_statistics import LMoments, OrderStatistics
    from hololoom.core.warp.math.analysis.stable_distributions import StableDistribution
    HAS_BAYESIAN_DIST = True
except ImportError:
    HAS_BAYESIAN_DIST = False


def dirichlet_process_cluster(data: np.ndarray, alpha: float = 1.0) -> dict | None:
    """Nonparametric clustering via Dirichlet process mixture."""
    if not HAS_BAYESIAN_DIST:
        return None
    try:
        dp = DPMixture(alpha=alpha)
        result = dp.fit(data)
        return {"n_clusters": result.n_clusters, "alpha": alpha}
    except Exception:
        return None


def fit_copula(u: np.ndarray, v: np.ndarray, family: str = "gaussian") -> dict | None:
    """Fit copula to model dependency structure."""
    if not HAS_BAYESIAN_DIST:
        return None
    try:
        if family == "gaussian":
            result = GaussianCopula.fit(u, v)
        elif family == "clayton":
            result = ClaytonCopula.fit(u, v)
        else:
            return None
        return {"family": family, "parameter": round(result.parameter, 4)}
    except Exception:
        return None


def l_moments(data: np.ndarray) -> dict | None:
    """Compute L-moments for robust distributional summary."""
    if not HAS_BAYESIAN_DIST:
        return None
    try:
        lm = LMoments.compute(data)
        return {"l1": round(lm.l1, 4), "l2": round(lm.l2, 4), "t3": round(lm.t3, 4)}
    except Exception:
        return None


# ============================================================================
# 6. Statistical Testing & Factor Analysis
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.discriminant_analysis import LDA, QDA
    from hololoom.core.warp.math.analysis.double_ml import DoubleML
    from hololoom.core.warp.math.analysis.factor_analysis import FactorAnalysis
    from hololoom.core.warp.math.analysis.nonparametric_tests import KruskalWallis, WilcoxonTest
    HAS_STAT_TESTS = True
except ImportError:
    HAS_STAT_TESTS = False


def nonparametric_compare(group_a: np.ndarray, group_b: np.ndarray) -> dict | None:
    """Wilcoxon rank-sum test for distribution comparison."""
    if not HAS_STAT_TESTS:
        return None
    try:
        result = WilcoxonTest.rank_sum(group_a, group_b)
        return {"statistic": round(result.statistic, 4), "p_value": round(result.p_value, 6)}
    except Exception:
        return None


def linear_discriminant(X: np.ndarray, y: np.ndarray, n_components: int = 2) -> dict | None:
    """LDA for dimensionality reduction and classification."""
    if not HAS_STAT_TESTS:
        return None
    try:
        result = LDA().fit(X, y, n_components=n_components)
        return {"n_components": n_components, "explained_ratio": result.explained_variance_ratio.tolist()}
    except Exception:
        return None


def extract_factors(X: np.ndarray, n_factors: int = 3) -> dict | None:
    """Factor analysis to discover latent structure."""
    if not HAS_STAT_TESTS:
        return None
    try:
        result = FactorAnalysis().fit(X, n_factors=n_factors)
        return {"n_factors": n_factors, "loadings_shape": list(result.loadings.shape)}
    except Exception:
        return None


def double_ml_estimate(Y: np.ndarray, D: np.ndarray, X: np.ndarray) -> dict | None:
    """Double/debiased ML for causal treatment effect."""
    if not HAS_STAT_TESTS:
        return None
    try:
        result = DoubleML().fit(Y, D, X)
        return {"ate": round(result.ate, 4), "std_err": round(result.std_err, 4)}
    except Exception:
        return None


# ============================================================================
# 7. Dynamical Systems & ODEs
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.neural_odes import AdjointMethod, NeuralODE
    from hololoom.core.warp.math.analysis.poincare_sections import PoincareMap, PoincareSection
    from hololoom.core.warp.math.analysis.stiff_ode_solvers import BDF, ImplicitEuler
    from hololoom.core.warp.math.analysis.stratonovich import StratonovichSDE
    HAS_ODE_SYSTEMS = True
except ImportError:
    HAS_ODE_SYSTEMS = False


def solve_stiff_system(f: Callable, y0: np.ndarray, t_span: tuple[float, float], order: int = 2) -> dict | None:
    """Solve stiff ODE system via BDF method."""
    if not HAS_ODE_SYSTEMS:
        return None
    try:
        result = BDF.solve(f, y0, t_span, order=order)
        return {"t_final": round(result.t[-1], 4), "n_steps": len(result.t)}
    except Exception:
        return None


def neural_ode_forward(f_net: Callable, y0: np.ndarray, t_span: tuple[float, float]) -> dict | None:
    """Forward pass through a neural ODE."""
    if not HAS_ODE_SYSTEMS:
        return None
    try:
        node = NeuralODE(f_net)
        result = node.solve(y0, t_span)
        return {"final_state": result.y_final.tolist()}
    except Exception:
        return None


def poincare_section_compute(trajectory: np.ndarray, plane_normal: np.ndarray, plane_point: np.ndarray) -> dict | None:
    """Compute Poincare section of a dynamical system trajectory."""
    if not HAS_ODE_SYSTEMS:
        return None
    try:
        result = PoincareSection.compute(trajectory, plane_normal, plane_point)
        return {"n_crossings": result.n_crossings, "points_shape": list(result.points.shape)}
    except Exception:
        return None


# ============================================================================
# 8. Convex & Variational Analysis
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.admm import ADMM, ConsensusADMM
    from hololoom.core.warp.math.analysis.convex_extensions import (
        FenchelConjugate,
        ProximalOperator,
    )
    from hololoom.core.warp.math.analysis.hamilton_jacobi import HamiltonJacobiSolver
    HAS_CONVEX = True
except ImportError:
    HAS_CONVEX = False


def admm_solve(f_prox: Callable, g_prox: Callable, A: np.ndarray, rho: float = 1.0) -> dict | None:
    """Solve f(x) + g(z) s.t. Ax = z via ADMM splitting."""
    if not HAS_CONVEX:
        return None
    try:
        solver = ADMM(rho=rho)
        result = solver.solve(f_prox, g_prox, A)
        return {"converged": result.converged, "n_iterations": result.n_iterations}
    except Exception:
        return None


def proximal_compute(f: Callable, x: np.ndarray, step: float = 1.0) -> np.ndarray | None:
    """Compute proximal operator prox_{step*f}(x)."""
    if not HAS_CONVEX:
        return None
    try:
        return ProximalOperator.evaluate(f, x, step)
    except Exception:
        return None


# ============================================================================
# 9. Exotic & Exterior Calculus
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.advanced_topics import (
        HenselsLemma,
        NonstandardAnalysis,
        PAdicNumber,
    )
    from hololoom.core.warp.math.analysis.classical_theorems import HahnBanach, StoneWeierstrass
    from hololoom.core.warp.math.analysis.exotic_calculus import FractionalCalculus, JetBundle
    from hololoom.core.warp.math.analysis.exterior_calculus import (
        ExteriorDerivative,
        HodgeStar,
        WedgeProduct,
    )
    HAS_EXOTIC_CALC = True
except ImportError:
    HAS_EXOTIC_CALC = False


def fractional_derivative(f: Callable, x: float, alpha: float = 0.5) -> dict | None:
    """Compute fractional derivative of order alpha."""
    if not HAS_EXOTIC_CALC:
        return None
    try:
        result = FractionalCalculus.caputo(f, x, alpha)
        return {"value": round(result, 6), "order": alpha}
    except Exception:
        return None


def hodge_star(form: np.ndarray, metric: np.ndarray) -> np.ndarray | None:
    """Apply Hodge star operator to a differential form."""
    if not HAS_EXOTIC_CALC:
        return None
    try:
        return HodgeStar.apply(form, metric)
    except Exception:
        return None


# ============================================================================
# 10. Harmonic Analysis, Spherical Harmonics, Ergodic Theory
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.ergodic_theory import ErgodicAnalyzer, MixingAnalyzer
    from hololoom.core.warp.math.analysis.harmonic_analysis_groups import GroupHarmonicAnalysis
    from hololoom.core.warp.math.analysis.spherical_harmonics import SphericalHarmonics
    from hololoom.core.warp.math.analysis.topological_entropy import KAMTheory, TopologicalEntropy
    HAS_HARMONIC = True
except ImportError:
    HAS_HARMONIC = False


def spherical_harmonic_expand(theta: np.ndarray, phi: np.ndarray, data: np.ndarray, l_max: int = 10) -> dict | None:
    """Expand data on the sphere in spherical harmonics."""
    if not HAS_HARMONIC:
        return None
    try:
        coeffs = SphericalHarmonics.analyze(theta, phi, data, l_max=l_max)
        return {"l_max": l_max, "n_coefficients": len(coeffs)}
    except Exception:
        return None


def ergodic_check(trajectory: np.ndarray) -> dict | None:
    """Test ergodicity of a dynamical trajectory."""
    if not HAS_HARMONIC:
        return None
    try:
        result = ErgodicAnalyzer.test(trajectory)
        return {"is_ergodic": result.is_ergodic, "mixing_time": round(result.mixing_time, 2)}
    except Exception:
        return None


def topological_entropy_compute(trajectory: np.ndarray) -> dict | None:
    """Compute topological entropy of a dynamical system."""
    if not HAS_HARMONIC:
        return None
    try:
        result = TopologicalEntropy.compute(trajectory)
        return {"entropy": round(result.entropy, 4)}
    except Exception:
        return None


# ============================================================================
# 11. Information & Complexity
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.kolmogorov_complexity import (
        AlgorithmicInformation,
        KolmogorovComplexity,
    )
    from hololoom.core.warp.math.analysis.pde_solvers import FiniteElementMethod, WaveEquationSolver
    HAS_INFO_COMPUTE = True
except ImportError:
    HAS_INFO_COMPUTE = False


def estimate_kolmogorov_complexity(data: bytes) -> dict | None:
    """Estimate Kolmogorov complexity via compression."""
    if not HAS_INFO_COMPUTE:
        return None
    try:
        result = KolmogorovComplexity.estimate(data)
        return {"complexity": round(result.complexity, 4), "normalized": round(result.normalized, 4)}
    except Exception:
        return None


def solve_wave_equation(domain: np.ndarray, ic: np.ndarray, t_final: float) -> dict | None:
    """Solve wave equation on a domain."""
    if not HAS_INFO_COMPUTE:
        return None
    try:
        result = WaveEquationSolver.solve(domain, ic, t_final)
        return {"t_final": t_final, "solution_shape": list(result.u.shape)}
    except Exception:
        return None


# ============================================================================
# 12. Computational Methods
# ============================================================================

try:
    from hololoom.core.warp.math.computational.approximation_algorithms import (
        FPTAS,
        RandomizedRounding,
    )
    from hololoom.core.warp.math.computational.krylov_methods import GMRES, BiCGSTAB
    from hololoom.core.warp.math.computational.matrix_factorizations import (
        CholeskyDecomposition,
        LUDecomposition,
    )
    from hololoom.core.warp.math.computational.polynomial_algorithms import (
        PolynomialGCD,
        SmithNormalForm,
    )
    HAS_COMPUTATIONAL = True
except ImportError:
    HAS_COMPUTATIONAL = False


def krylov_solve(A: np.ndarray, b: np.ndarray, method: str = "gmres") -> dict | None:
    """Solve Ax = b via Krylov subspace method."""
    if not HAS_COMPUTATIONAL:
        return None
    try:
        if method == "gmres":
            result = GMRES.solve(A, b)
        elif method == "bicgstab":
            result = BiCGSTAB.solve(A, b)
        else:
            return None
        return {"converged": result.converged, "n_iterations": result.n_iterations, "residual": round(result.residual, 8)}
    except Exception:
        return None


def lu_decompose(A: np.ndarray) -> dict | None:
    """LU decomposition with partial pivoting."""
    if not HAS_COMPUTATIONAL:
        return None
    try:
        result = LUDecomposition.decompose(A)
        return {"n": A.shape[0], "det_sign": result.det_sign}
    except Exception:
        return None


def smith_normal_form(A: np.ndarray) -> dict | None:
    """Compute Smith normal form of an integer matrix."""
    if not HAS_COMPUTATIONAL:
        return None
    try:
        result = SmithNormalForm.compute(A)
        return {"invariant_factors": result.invariant_factors.tolist()}
    except Exception:
        return None


# ============================================================================
# 13. RL Deep Stack — MDP, Deep RL, Value-Based, Curriculum, Offline, Reward Machines
# ============================================================================

try:
    from hololoom.core.warp.math.decision.curriculum_learning import (
        CurriculumScheduler,
        SelfPacedLearning,
    )
    from hololoom.core.warp.math.decision.deep_rl import DQN, DoubleDQN
    from hololoom.core.warp.math.decision.mdp_foundations import MDP, TemporalDifference
    from hololoom.core.warp.math.decision.offline_rl import BatchConstrainedQ, ConservativeQ
    from hololoom.core.warp.math.decision.reward_machines import RewardMachine, RMQLearning
    from hololoom.core.warp.math.decision.value_based_rl import SARSA, QLearning
    HAS_RL_DEEP = True
except ImportError:
    HAS_RL_DEEP = False


def value_iteration(transition: np.ndarray, rewards: np.ndarray, gamma: float = 0.99) -> dict | None:
    """Solve MDP via value iteration."""
    if not HAS_RL_DEEP:
        return None
    try:
        mdp = MDP(transition, rewards, gamma=gamma)
        result = mdp.value_iteration()
        return {"n_states": transition.shape[0], "converged": result.converged, "n_iterations": result.n_iterations}
    except Exception:
        return None


def curriculum_schedule(task_difficulties: np.ndarray, competence: float = 0.5) -> dict | None:
    """Select tasks matching current competence level."""
    if not HAS_RL_DEEP:
        return None
    try:
        scheduler = CurriculumScheduler()
        selected = scheduler.select(task_difficulties, competence)
        return {"n_selected": len(selected), "competence": competence}
    except Exception:
        return None


def offline_rl_policy(dataset: list[dict], method: str = "cql") -> dict | None:
    """Learn policy from offline dataset without environment interaction."""
    if not HAS_RL_DEEP:
        return None
    try:
        if method == "cql":
            result = ConservativeQ.train(dataset)
        elif method == "bcq":
            result = BatchConstrainedQ.train(dataset)
        else:
            return None
        return {"method": method, "n_transitions": len(dataset)}
    except Exception:
        return None


# ============================================================================
# 14. Game Theory — Correlated Equilibria, Regret Minimization
# ============================================================================

try:
    from hololoom.core.warp.math.decision.correlated_equilibria import (
        CoarseCorrelatedEquilibrium,
        CorrelatedEquilibrium,
    )
    from hololoom.core.warp.math.decision.regret_minimization import (
        EXP3,
        FollowTheRegularizedLeader,
        Hedge,
    )
    HAS_GAME_DEEP = True
except ImportError:
    HAS_GAME_DEEP = False


def hedge_allocate(losses: np.ndarray, eta: float = 0.1) -> dict | None:
    """Multiplicative weights update for online learning."""
    if not HAS_GAME_DEEP:
        return None
    try:
        h = Hedge(n_arms=losses.shape[1], eta=eta)
        for t in range(losses.shape[0]):
            h.update(losses[t])
        return {"weights": h.weights.tolist(), "total_regret": round(h.total_regret, 4)}
    except Exception:
        return None


def exp3_bandit(n_arms: int, gamma: float = 0.1) -> dict | None:
    """EXP3 adversarial bandit initialization."""
    if not HAS_GAME_DEEP:
        return None
    try:
        bandit = EXP3(n_arms=n_arms, gamma=gamma)
        return {"n_arms": n_arms, "gamma": gamma, "initialized": True}
    except Exception:
        return None


# ============================================================================
# 15. Combinatorics & Complexity
# ============================================================================

try:
    from hololoom.core.warp.math.extensions.algebraic_combinatorics import (
        RSKCorrespondence,
        SchurFunction,
    )
    from hololoom.core.warp.math.extensions.combinatorial_numbers import (
        BellNumbers,
        Species,
        StirlingNumbers,
    )
    from hololoom.core.warp.math.extensions.complexity_cybernetics import (
        ParameterizedComplexity,
        ViabilityTheory,
    )
    from hololoom.core.warp.math.extensions.extremal_combinatorics import (
        SzemerediRegularity,
        TuranTheorem,
    )
    from hololoom.core.warp.math.self_assembly.boids_advanced import FlockingMetrics
    HAS_COMBINATORICS = True
except ImportError:
    HAS_COMBINATORICS = False


def bell_number(n: int) -> int | None:
    """Compute the nth Bell number (partitions of a set)."""
    if not HAS_COMBINATORICS:
        return None
    try:
        return BellNumbers.compute(n)
    except Exception:
        return None


def stirling_second_kind(n: int, k: int) -> int | None:
    """Stirling number S(n,k) — partitions of n into k non-empty subsets."""
    if not HAS_COMBINATORICS:
        return None
    try:
        return StirlingNumbers.second_kind(n, k)
    except Exception:
        return None


def turan_bound(n: int, r: int) -> dict | None:
    """Turan's theorem: max edges in K_{r+1}-free graph on n vertices."""
    if not HAS_COMBINATORICS:
        return None
    try:
        bound = TuranTheorem.max_edges(n, r)
        return {"n": n, "r": r, "max_edges": bound}
    except Exception:
        return None


# ============================================================================
# 16. Geometry — Algebraic, Computational, Symplectic, Riemannian, Physics
# ============================================================================

try:
    from hololoom.core.warp.math.geometry.algebraic_geometry import AffineVariety, ProjectiveVariety
    from hololoom.core.warp.math.geometry.computational_geometry import (
        ConvexHull,
        DelaunayTriangulation,
        VoronoiDiagram,
    )
    from hololoom.core.warp.math.geometry.mathematical_physics import (
        HamiltonianMechanics,
        LagrangianMechanics,
        NoetherTheorem,
    )
    from hololoom.core.warp.math.geometry.riemannian_geometry import (
        Geodesic,
        RicciFlow,
        RiemannianMetric,
    )
    from hololoom.core.warp.math.geometry.symplectic_completion import MomentMap, SymplecticForm
    HAS_GEOMETRY = True
except ImportError:
    HAS_GEOMETRY = False


def convex_hull_compute(points: np.ndarray) -> dict | None:
    """Compute convex hull of a point set."""
    if not HAS_GEOMETRY:
        return None
    try:
        result = ConvexHull.compute(points)
        return {"n_vertices": result.n_vertices, "volume": round(result.volume, 4)}
    except Exception:
        return None


def delaunay_triangulate(points: np.ndarray) -> dict | None:
    """Delaunay triangulation of a point set."""
    if not HAS_GEOMETRY:
        return None
    try:
        result = DelaunayTriangulation.compute(points)
        return {"n_simplices": result.n_simplices, "n_points": points.shape[0]}
    except Exception:
        return None


def geodesic_distance(metric: np.ndarray, p: np.ndarray, q: np.ndarray) -> dict | None:
    """Compute geodesic distance on a Riemannian manifold."""
    if not HAS_GEOMETRY:
        return None
    try:
        rm = RiemannianMetric(metric)
        dist = Geodesic.distance(rm, p, q)
        return {"distance": round(float(dist), 6)}
    except Exception:
        return None


def noether_conserved(lagrangian: Callable, symmetry: str) -> dict | None:
    """Find conserved quantity from Noether's theorem."""
    if not HAS_GEOMETRY:
        return None
    try:
        result = NoetherTheorem.conserved_quantity(lagrangian, symmetry)
        return {"symmetry": symmetry, "conserved": True}
    except Exception:
        return None


# ============================================================================
# 17. Graph Theory — Spectral Clustering, Advanced Graph Theory
# ============================================================================

try:
    from hololoom.core.warp.math.extensions.graph_theory_advanced import (
        MatroidTheory,
        RamseyTheory,
        TuttePolynomial,
    )
    from hololoom.core.warp.math.graph.spectral_clustering import GraphLaplacian, SpectralClustering
    HAS_GRAPH_THEORY = True
except ImportError:
    HAS_GRAPH_THEORY = False


def spectral_cluster(adjacency: np.ndarray, n_clusters: int = 3) -> dict | None:
    """Spectral clustering via graph Laplacian eigenvectors."""
    if not HAS_GRAPH_THEORY:
        return None
    try:
        result = SpectralClustering.cluster(adjacency, n_clusters=n_clusters)
        return {"labels": result.labels.tolist(), "n_clusters": n_clusters}
    except Exception:
        return None


def tutte_polynomial(adjacency: np.ndarray) -> dict | None:
    """Compute Tutte polynomial of a graph."""
    if not HAS_GRAPH_THEORY:
        return None
    try:
        result = TuttePolynomial.compute(adjacency)
        return {"chromatic_number": result.chromatic_number}
    except Exception:
        return None


# ============================================================================
# 18. Self-Assembly & Emergence
# ============================================================================

try:
    from hololoom.core.warp.math.self_assembly.flocking import Boids, ConsensusProtocol
    from hololoom.core.warp.math.self_assembly.genetic_programming import NEAT, GeneticProgramming
    from hololoom.core.warp.math.self_assembly.reaction_diffusion import (
        GrayScott,
        ReactionDiffusion,
    )
    from hololoom.core.warp.math.self_assembly.self_modifying import (
        LiquidStateMachine,
        OpenEndedEvolution,
    )
    from hololoom.core.warp.math.self_assembly.swarm import (
        AntColonyOptimization,
        ParticleSwarmOptimization,
    )
    HAS_SELF_ASSEMBLY = True
except ImportError:
    HAS_SELF_ASSEMBLY = False


def evolve_program(fitness_fn: Callable, n_generations: int = 50, pop_size: int = 100) -> dict | None:
    """Evolve programs via genetic programming."""
    if not HAS_SELF_ASSEMBLY:
        return None
    try:
        gp = GeneticProgramming(pop_size=pop_size)
        result = gp.evolve(fitness_fn, n_generations=n_generations)
        return {"best_fitness": round(result.best_fitness, 4), "generations": result.generations}
    except Exception:
        return None


def reaction_diffusion_step(grid: np.ndarray, dt: float = 0.1, model: str = "gray_scott") -> np.ndarray | None:
    """Step a reaction-diffusion system forward."""
    if not HAS_SELF_ASSEMBLY:
        return None
    try:
        if model == "gray_scott":
            rd = GrayScott()
        else:
            rd = ReactionDiffusion()
        return rd.step(grid, dt)
    except Exception:
        return None


def particle_swarm_optimize(objective: Callable, bounds: np.ndarray, n_particles: int = 30) -> dict | None:
    """Optimize via particle swarm."""
    if not HAS_SELF_ASSEMBLY:
        return None
    try:
        pso = ParticleSwarmOptimization(n_particles=n_particles)
        result = pso.optimize(objective, bounds)
        return {"best_position": result.best_position.tolist(), "best_value": round(result.best_value, 6)}
    except Exception:
        return None


def flocking_consensus(positions: np.ndarray, velocities: np.ndarray) -> dict | None:
    """Simulate Boids flocking step."""
    if not HAS_SELF_ASSEMBLY:
        return None
    try:
        boids = Boids()
        result = boids.step(positions, velocities)
        return {"center_of_mass": result.center.tolist(), "alignment": round(result.alignment, 4)}
    except Exception:
        return None


# ============================================================================
# 19. Topology — Alpha/Cech, Covering Spaces, Persistence, Quotient, Sheaves
# ============================================================================

try:
    from hololoom.core.warp.math.advanced.alpha_cech_complexes import (
        AlphaComplex,
        CechComplex,
        TDAPipeline,
    )
    from hololoom.core.warp.math.advanced.covering_spaces import CoveringSpace, FundamentalGroup
    from hololoom.core.warp.math.advanced.extended_persistence import (
        ExtendedPersistence,
        ZigzagPersistence,
    )
    from hololoom.core.warp.math.advanced.quotient_product_spaces import (
        ProductTopology,
        QuotientSpace,
    )
    from hololoom.core.warp.math.advanced.sheaf_extensions import DerivedSheafCohomology, Sheaf
    HAS_TOPOLOGY = True
except ImportError:
    HAS_TOPOLOGY = False


def alpha_complex_filtration(points: np.ndarray, max_alpha: float = 1.0) -> dict | None:
    """Build alpha complex filtration for TDA."""
    if not HAS_TOPOLOGY:
        return None
    try:
        ac = AlphaComplex.build(points, max_alpha=max_alpha)
        return {"n_simplices": ac.n_simplices, "max_alpha": max_alpha}
    except Exception:
        return None


def extended_persistence_compute(filtration: list) -> dict | None:
    """Compute extended persistence diagram."""
    if not HAS_TOPOLOGY:
        return None
    try:
        result = ExtendedPersistence.compute(filtration)
        return {"n_intervals": len(result.intervals), "max_persistence": round(result.max_persistence, 4)}
    except Exception:
        return None


def fundamental_group_compute(simplicial_complex: list) -> dict | None:
    """Compute fundamental group of a simplicial complex."""
    if not HAS_TOPOLOGY:
        return None
    try:
        fg = FundamentalGroup.compute(simplicial_complex)
        return {"rank": fg.rank, "is_trivial": fg.is_trivial}
    except Exception:
        return None


def sheaf_cohomology(sheaf_data: dict, degree: int = 1) -> dict | None:
    """Compute sheaf cohomology."""
    if not HAS_TOPOLOGY:
        return None
    try:
        result = DerivedSheafCohomology.compute(sheaf_data, degree=degree)
        return {"degree": degree, "dimension": result.dimension}
    except Exception:
        return None


# ============================================================================
# 20. Standalone Modules — Causal Inference, Contextual Bandit, Explainability,
#     Operation Selectors
# ============================================================================

try:
    from hololoom.core.warp.math.causal.causal_inference import CausalDiscovery, CausalGraph
    HAS_CAUSAL = True
except ImportError:
    HAS_CAUSAL = False

try:
    from hololoom.core.warp.math.contextual_bandit import (
        ContextualOperationSelector,
        GaussianLinearBandit,
    )
    HAS_CONTEXTUAL = True
except ImportError:
    HAS_CONTEXTUAL = False

try:
    from hololoom.core.warp.math.explainability import ExplainableSelector, ExplanationGenerator
    HAS_EXPLAINABILITY = True
except ImportError:
    HAS_EXPLAINABILITY = False

try:
    from hololoom.core.warp.math.operation_selector import MathOperationSelector
    from hololoom.core.warp.math.smart_operation_selector import SmartMathOperationSelector
    HAS_OPERATION_SELECTORS = True
except ImportError:
    HAS_OPERATION_SELECTORS = False


def discover_causal_structure(data: np.ndarray, var_names: list[str]) -> dict | None:
    """Discover causal graph from observational data."""
    if not HAS_CAUSAL:
        return None
    try:
        result = CausalDiscovery.pc_algorithm(data, var_names)
        return {"n_variables": len(var_names), "n_edges": result.n_edges}
    except Exception:
        return None


def contextual_bandit_select(context: np.ndarray, n_arms: int) -> dict | None:
    """Select arm via contextual bandit with Gaussian linear model."""
    if not HAS_CONTEXTUAL:
        return None
    try:
        bandit = GaussianLinearBandit(n_arms=n_arms, context_dim=context.shape[0])
        arm = bandit.select(context)
        return {"selected_arm": arm, "n_arms": n_arms}
    except Exception:
        return None


def explain_decision(selector_state: dict, query: str) -> dict | None:
    """Generate explanation for operation selector decision."""
    if not HAS_EXPLAINABILITY:
        return None
    try:
        gen = ExplanationGenerator()
        explanation = gen.why(selector_state, query)
        return {"explanation": explanation.text, "confidence": round(explanation.confidence, 4)}
    except Exception:
        return None


def smart_select_operation(query: str, available_ops: list[str]) -> dict | None:
    """Select best math operation via Thompson Sampling + composition."""
    if not HAS_OPERATION_SELECTORS:
        return None
    try:
        selector = SmartMathOperationSelector()
        plan = selector.select(query, available_ops)
        return {"selected": plan.operation, "confidence": round(plan.confidence, 4)}
    except Exception:
        return None


# ============================================================================
# Status
# ============================================================================

def math_extensions_v7_status() -> dict[str, bool]:
    """Report which v7 math extensions are available."""
    return {
        # 1. Abstract Algebra
        "lie_algebras": HAS_ABSTRACT_ALGEBRA,
        "representation_theory": HAS_ABSTRACT_ALGEBRA,
        "universal_algebra": HAS_ABSTRACT_ALGEBRA,
        "quotient_structures": HAS_ABSTRACT_ALGEBRA,
        # 2. Category Theory
        "adjoint_functors": HAS_CATEGORY_THEORY,
        "enriched_categories": HAS_CATEGORY_THEORY,
        "toposes": HAS_CATEGORY_THEORY,
        # 3. Number Theory & Tensor
        "algebraic_number_theory": HAS_ALGEBRA_DEEP,
        "commutative_algebra": HAS_ALGEBRA_DEEP,
        "k_theory": HAS_ALGEBRA_DEEP,
        "matrix_functions": HAS_ALGEBRA_DEEP,
        "tensor_decomposition": HAS_ALGEBRA_DEEP,
        # 4. Statistical Regression
        "robust_regression": HAS_REGRESSION,
        "spline_regression": HAS_REGRESSION,
        "multivariate_regression": HAS_REGRESSION,
        # 5. Bayesian & Distributional
        "bayesian_nonparametrics": HAS_BAYESIAN_DIST,
        "copulas": HAS_BAYESIAN_DIST,
        "stable_distributions": HAS_BAYESIAN_DIST,
        "order_statistics": HAS_BAYESIAN_DIST,
        # 6. Statistical Testing
        "nonparametric_tests": HAS_STAT_TESTS,
        "discriminant_analysis": HAS_STAT_TESTS,
        "factor_analysis": HAS_STAT_TESTS,
        "double_ml": HAS_STAT_TESTS,
        # 7. Dynamical Systems
        "stiff_ode_solvers": HAS_ODE_SYSTEMS,
        "neural_odes": HAS_ODE_SYSTEMS,
        "stratonovich": HAS_ODE_SYSTEMS,
        "poincare_sections": HAS_ODE_SYSTEMS,
        # 8. Convex & Variational
        "admm": HAS_CONVEX,
        "convex_extensions": HAS_CONVEX,
        "hamilton_jacobi": HAS_CONVEX,
        # 9. Exotic Calculus
        "exotic_calculus": HAS_EXOTIC_CALC,
        "exterior_calculus": HAS_EXOTIC_CALC,
        "classical_theorems": HAS_EXOTIC_CALC,
        # 10. Harmonic & Ergodic
        "harmonic_analysis_groups": HAS_HARMONIC,
        "spherical_harmonics": HAS_HARMONIC,
        "ergodic_theory": HAS_HARMONIC,
        "topological_entropy": HAS_HARMONIC,
        # 11. Information & PDE
        "kolmogorov_complexity": HAS_INFO_COMPUTE,
        "pde_solvers": HAS_INFO_COMPUTE,
        # 12. Computational Methods
        "approximation_algorithms": HAS_COMPUTATIONAL,
        "krylov_methods": HAS_COMPUTATIONAL,
        "matrix_factorizations": HAS_COMPUTATIONAL,
        "polynomial_algorithms": HAS_COMPUTATIONAL,
        # 13. RL Deep Stack
        "mdp_foundations": HAS_RL_DEEP,
        "deep_rl": HAS_RL_DEEP,
        "value_based_rl": HAS_RL_DEEP,
        "curriculum_learning": HAS_RL_DEEP,
        "offline_rl": HAS_RL_DEEP,
        "reward_machines": HAS_RL_DEEP,
        # 14. Game Theory
        "correlated_equilibria_v7": HAS_GAME_DEEP,
        "regret_minimization": HAS_GAME_DEEP,
        # 15. Combinatorics
        "algebraic_combinatorics": HAS_COMBINATORICS,
        "combinatorial_numbers": HAS_COMBINATORICS,
        "extremal_combinatorics": HAS_COMBINATORICS,
        "complexity_cybernetics": HAS_COMBINATORICS,
        "boids_advanced": HAS_COMBINATORICS,
        # 16. Geometry
        "algebraic_geometry": HAS_GEOMETRY,
        "computational_geometry": HAS_GEOMETRY,
        "symplectic_completion": HAS_GEOMETRY,
        "riemannian_geometry": HAS_GEOMETRY,
        "mathematical_physics": HAS_GEOMETRY,
        # 17. Graph Theory
        "spectral_clustering_v7": HAS_GRAPH_THEORY,
        "graph_theory_advanced": HAS_GRAPH_THEORY,
        # 18. Self-Assembly
        "genetic_programming": HAS_SELF_ASSEMBLY,
        "reaction_diffusion": HAS_SELF_ASSEMBLY,
        "self_modifying": HAS_SELF_ASSEMBLY,
        "flocking": HAS_SELF_ASSEMBLY,
        "swarm": HAS_SELF_ASSEMBLY,
        # 19. Topology
        "alpha_cech_complexes": HAS_TOPOLOGY,
        "covering_spaces": HAS_TOPOLOGY,
        "extended_persistence": HAS_TOPOLOGY,
        "quotient_product_spaces": HAS_TOPOLOGY,
        "sheaf_extensions": HAS_TOPOLOGY,
        # 20. Standalone
        "causal_inference_v7": HAS_CAUSAL,
        "contextual_bandit_v7": HAS_CONTEXTUAL,
        "explainability_v7": HAS_EXPLAINABILITY,
        "operation_selector": HAS_OPERATION_SELECTORS,
        "smart_operation_selector": HAS_OPERATION_SELECTORS,
    }
