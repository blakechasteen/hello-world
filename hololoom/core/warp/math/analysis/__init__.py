"""
Analysis - Complete Mathematical Foundation
============================================

Real Analysis: Metric spaces, sequences, continuity, differentiation, integration
Complex Analysis: Holomorphic functions, residues, conformal maps
Functional Analysis: Banach/Hilbert spaces, operators, spectral theory
Measure Theory: Sigma-algebras, measures, Lebesgue integration
Fourier & Harmonic: Fourier transforms, wavelets, time-frequency analysis
Stochastic Calculus: Brownian motion, Ito calculus, SDEs
Advanced Topics: Microlocal, nonstandard, p-adic analysis
Numerical Analysis: Root finding, ODEs, optimization, interpolation
Probability Theory: Random variables, distributions, inference, Markov chains
Distribution Theory: Schwartz functions, Dirac delta, Green's functions

Kalman Filtering: Linear/nonlinear state estimation, RTS smoothing
Extreme Value Theory: GEV, GPD, tail risk, return levels
Time Series Models: ARIMA, exponential smoothing, autocorrelation
Changepoint Detection: CUSUM, PELT, Bayesian online detection

Stability Analysis: Lyapunov exponents, fixed points, linear stability
Randomized Linear Algebra: Fast approximate SVD, JL projections, Nystrom

Hidden Markov Models: Forward-backward, Viterbi, Baum-Welch, Gaussian HMM
Fokker-Planck Equation: Probability density evolution, Kramers escape
Bifurcation Analysis: Critical transitions, Hopf/saddle-node detection
Kernel Regression: Nadaraya-Watson, local linear, KDE, RKHS

Stiff ODE Solvers: BDF, implicit Euler, trapezoidal rule
Neural ODEs: Continuous-depth models, adjoint method, CNF
Poincare Sections: Phase space crossings, return maps, winding number
Survival Analysis: Kaplan-Meier, Cox PH, Nelson-Aalen, log-rank test
Mixture Models: Gaussian EM, Bayesian GMM with Dirichlet prior
Robust Regression: Huber, RANSAC, Theil-Sen
Nonparametric Tests: Wilcoxon, Mann-Whitney, Kruskal-Wallis, Friedman
Ergodic Theory: Time averages, mixing rates, Poincare recurrence
Wasserstein Distances: Optimal transport, Sinkhorn, barycenters, sliced
Bayesian Nonparametrics: Dirichlet process, CRP, DP mixture
Copulas: Gaussian, Clayton, Frank copulas, Kendall's tau

ADMM: Alternating direction method of multipliers, consensus optimization
Spherical Harmonics: Y_l^m evaluation, decomposition, reconstruction
Factor Analysis: EM algorithm, varimax rotation, communalities
Discriminant Analysis: Fisher's LDA, QDA classification
Bootstrap: Confidence intervals, permutation tests, jackknife
Streaming: Count-Min Sketch, HyperLogLog, reservoir sampling
Density Estimation: Histogram, KDE, wavelet shrinkage

Spline Regression: LOESS, B-spline basis, VAR, regression diagnostics
Double ML: Debiased causal inference, synthetic control
Kolmogorov Complexity: Compression-based complexity, NCD, algorithmic information
Convex Extensions: Fenchel conjugate, Moreau envelope, proximal operators
Topological Entropy: KAM theory, topological entropy, full Lyapunov spectrum

Complete 43-module analysis suite for rigorous mathematics in AI.
"""

from .admm import (
    ADMM,
    ADMMResult,
    ConsensusADMM,
    soft_threshold,
)
from .advanced_topics import (
    HenselsLemma,
    Hyperreal,
    NonstandardAnalysis,
    PAdicNumber,
    PseudodifferentialOperator,
    WaveFrontSet,
)
from .bayesian_nonparametrics import (
    ChineseRestaurantProcess,
    DirichletProcess,
    DPMixture,
    DPMixtureResult,
    expected_clusters,
)
from .bifurcation_analysis import (
    BifurcationAnalyzer,
    BifurcationDiagram,
    BifurcationPoint,
    HopfBifurcation,
    pitchfork_normal_form,
    saddle_node_normal_form,
    transcritical_normal_form,
)
from .bootstrap import (
    Bootstrap,
    Jackknife,
    PermutationTest,
)
from .changepoint_detection import (
    CUSUM,
    PELT,
    BayesianChangepoint,
    Changepoint,
    ChangepointResult,
    segment_statistics,
)
from .chaos_dynamical_systems import (
    AttractorResult,
    PredictabilityAnalyzer,
    StrangeAttractorDetector,
    analyze_attractor,
    henon_map,
    lorenz_system,
    rossler_system,
)
from .classical_theorems import (
    ArzelaAscoli,
    HaarMeasure,
    HahnBanach,
    StoneWeierstrass,
    WeakTopology,
)
from .complex_analysis import (
    AnalyticContinuation,
    ComplexFunction,
    ConformalMapper,
    ContourIntegrator,
    ResidueCalculator,
    SeriesExpansion,
)
from .convex_extensions import (
    ConjugateResult,
    FenchelConjugate,
    MoreauEnvelope,
    ProximalOperator,
    ProxResult,
)
from .copulas import (
    ClaytonCopula,
    CopulaResult,
    FrankCopula,
    GaussianCopula,
    kendall_tau,
)
from .density_estimation import (
    DensityResult,
    HistogramEstimator,
    KDEEstimator,
    WaveletDensity,
)
from .discriminant_analysis import (
    LDA,
    QDA,
    LDAResult,
)
from .distribution_theory import (
    Distribution,
    DistributionFourier,
    GreenFunction,
    SchwartzFunction,
    StandardDistributions,
    WeakDerivative,
)
from .double_ml import (
    DMLResult,
    DoubleML,
    SyntheticControl,
    SyntheticResult,
)
from .ergodic_theory import (
    ErgodicAnalyzer,
    ErgodicResult,
    MixingAnalyzer,
    MixingResult,
    RecurrenceAnalyzer,
    RecurrenceResult,
    is_ergodic,
)
from .exotic_calculus import (
    FractionalCalculus,
    JetBundle,
    JetData,
    PrincipalUltrafilter,
    UltrafilterConvergence,
)
from .exterior_calculus import (
    DeRhamCohomology,
    ExteriorDerivative,
    HodgeStar,
    WedgeProduct,
)
from .extreme_value_distributions import (
    EVDResult,
    GeneralizedExtremeValue,
    GEVParams,
    GumbelDistribution,
    PeaksOverThreshold,
    block_maxima,
)
from .factor_analysis import (
    FactorAnalysis,
    FactorResult,
)
from .fokker_planck import (
    FokkerPlanckSolver,
    FPESolution,
)
from .fourier_harmonic import (
    FourierSeries,
    FourierTransform,
    MultitaperSpectral,
    SpectralTimeSeries,
    TimeFrequencyAnalysis,
    WaveletTransform,
)
from .functional_analysis import (
    BoundedOperator,
    CompactOperator,
    HilbertSpace,
    NormedSpace,
    SobolevSpace,
    SpectralAnalyzer,
)
from .hamilton_jacobi import (
    HamiltonJacobiSolver,
    HJResult,
)
from .harmonic_analysis_groups import (
    GroupHarmonicAnalysis,
    RiemannSurface,
    RiemannSurfaceData,
)
from .hidden_markov_models import (
    GaussianHMM,
    GaussianHMMParams,
    HiddenMarkovModel,
    HMMParams,
    HMMResult,
)
from .kalman_filtering import (
    ExtendedKalmanFilter,
    KalmanFilter,
    KalmanState,
    UnscentedKalmanFilter,
    steady_state_gain,
)
from .kernel_regression import (
    RKHS,
    KernelDensityEstimation,
    KernelRegressionResult,
    LocalLinearRegression,
    NadarayaWatson,
    epanechnikov_kernel,
    rbf_kernel,
    tricube_kernel,
)
from .kolmogorov_complexity import (
    AlgorithmicInformation,
    BerryParadox,
    ComplexityResult,
    KolmogorovComplexity,
)
from .measure_theory import (
    ConvergenceTheorems,
    LebesgueIntegrator,
    LebesgueMeasure,
    MeasurableFunction,
    Measure,
    SigmaAlgebra,
)
from .mixture_models import (
    BayesianGaussianMixture,
    GaussianMixture,
    MixtureResult,
)

# Multivariate Regression
from .multivariate_regression import (
    MultivariateRegression,
    MultivariateResult,
)
from .neural_odes import (
    AdjointMethod,
    AdjointResult,
    CNFResult,
    ContinuousNormalizing,
    NeuralODE,
    NeuralODEResult,
)
from .nonparametric_tests import (
    FriedmanTest,
    KruskalWallis,
    MannWhitneyU,
    WilcoxonTest,
    multiple_testing_correction,
)
from .numerical_analysis import (
    Interpolation,
    NumericalLinearAlgebra,
    NumericalOptimization,
    ODESolution,
    ODESolver,
    RootFinder,
)

# Order Statistics & L-Moments
from .order_statistics import (
    LMoments,
    OrderStatistics,
)
from .pde_solvers import (
    FEMResult,
    FiniteElementMethod,
    LaplaceEquationSolver,
    LaplaceResult,
    WaveEquationSolver,
    WaveResult,
)
from .poincare_sections import (
    PoincareMap,
    PoincareSection,
    ReturnMapResult,
    SectionResult,
    winding_number,
)
from .probability_theory import (
    BayesianInference,
    CommonDistributions,
    HypothesisTesting,
    LimitTheorems,
    MarkovChain,
    MaximumLikelihoodEstimation,
    ProbabilitySpace,
    RandomVariable,
)
from .randomized_linear_algebra import (
    JohnsonLindenstrauss,
    RandomizedNystrom,
    RandomizedSVD,
    TruncatedSVDResult,
    jl_project,
    randomized_svd,
)
from .real_analysis import (
    BarbashinKrasovskii,
    ContinuityChecker,
    Differentiator,
    LaSalleInvariance,
    MetricSpace,
    RiemannIntegrator,
    SequenceAnalyzer,
)
from .robust_regression import (
    HuberRegression,
    RANSACRegression,
    RegressionResult,
    TheilSenEstimator,
)
from .spherical_harmonics import (
    AssociatedLegendre,
    SphericalHarmonics,
)
from .spline_regression import (
    LOESSRegression,
    RegressionDiagnostics,
    SplineRegression,
    SplineResult,
    VARModel,
    VARResult,
)
from .stability_analysis import (
    FixedPoint,
    FixedPointFinder,
    LinearStability,
    LyapunovExponents,
    LyapunovResult,
    find_equilibria,
    is_chaotic,
    stability_classification,
)

# Stable Distributions & Advanced Statistical Methods
from .stable_distributions import (
    MDS,
    CanonicalCorrelation,
    CCAResult,
    EmpiricalBayes,
    KernelDensity1D,
    LocalDP,
    RankTest,
    RenyiDP,
    StableDistribution,
    VariationalInference,
)
from .stiff_ode_solvers import (
    BDF,
    ImplicitEuler,
    ODEResult,
    TrapezoidalRule,
)
from .stochastic_calculus import (
    BrownianMotion,
    ItoIntegrator,
    ItosLemma,
    MartingaleAnalyzer,
    SDEResult,
    StochasticDifferentialEquation,
)
from .stratonovich import (
    SDEResult as StratonovichSDEResult,
)
from .stratonovich import (
    StratonovichSDE,
)
from .streaming import (
    CountMinSketch,
    ExponentialHistogram,
    HyperLogLog,
    ReservoirSampling,
)
from .survival_analysis import (
    CoxProportionalHazards,
    CoxResult,
    KaplanMeier,
    NelsonAalen,
    SurvivalResult,
    log_rank_test,
)
from .time_series_models import (
    ARIMA,
    ARIMAParams,
    AutoCorrelation,
    ExponentialSmoothing,
    ForecastResult,
)
from .topological_entropy import (
    KAMResult,
    KAMTheory,
    LyapunovSpectrum,
    LyapunovSpectrumResult,
    TopologicalEntropy,
    TopologicalEntropyResult,
)
from .wasserstein import (
    SinkhornDivergence,
    SlicedWasserstein,
    TransportResult,
    WassersteinBarycenter,
    WassersteinDistance,
)

__all__ = [
    # Real Analysis
    "MetricSpace",
    "SequenceAnalyzer",
    "ContinuityChecker",
    "Differentiator",
    "RiemannIntegrator",
    "LaSalleInvariance",
    "BarbashinKrasovskii",
    # Complex Analysis
    "ComplexFunction",
    "ContourIntegrator",
    "ResidueCalculator",
    "ConformalMapper",
    "SeriesExpansion",
    "AnalyticContinuation",
    # Functional Analysis
    "NormedSpace",
    "HilbertSpace",
    "BoundedOperator",
    "SpectralAnalyzer",
    "SobolevSpace",
    "CompactOperator",
    # Measure Theory
    "SigmaAlgebra",
    "Measure",
    "LebesgueMeasure",
    "MeasurableFunction",
    "LebesgueIntegrator",
    "ConvergenceTheorems",
    # Fourier & Harmonic
    "FourierTransform",
    "FourierSeries",
    "WaveletTransform",
    "TimeFrequencyAnalysis",
    "SpectralTimeSeries",
    "MultitaperSpectral",
    # Stochastic Calculus
    "BrownianMotion",
    "MartingaleAnalyzer",
    "ItoIntegrator",
    "ItosLemma",
    "StochasticDifferentialEquation",
    "SDEResult",
    # Advanced Topics
    "WaveFrontSet",
    "PseudodifferentialOperator",
    "Hyperreal",
    "NonstandardAnalysis",
    "PAdicNumber",
    "HenselsLemma",
    # Numerical Analysis
    "RootFinder",
    "NumericalLinearAlgebra",
    "ODESolver",
    "ODESolution",
    "Interpolation",
    "NumericalOptimization",
    # Probability Theory
    "ProbabilitySpace",
    "RandomVariable",
    "CommonDistributions",
    "LimitTheorems",
    "MaximumLikelihoodEstimation",
    "BayesianInference",
    "HypothesisTesting",
    "MarkovChain",
    # Distribution Theory
    "SchwartzFunction",
    "Distribution",
    "StandardDistributions",
    "DistributionFourier",
    "GreenFunction",
    "WeakDerivative",
    # Kalman Filtering
    "KalmanState",
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "steady_state_gain",
    # Extreme Value Theory
    "GEVParams",
    "EVDResult",
    "GeneralizedExtremeValue",
    "GumbelDistribution",
    "PeaksOverThreshold",
    "block_maxima",
    # Time Series Models
    "ARIMAParams",
    "ForecastResult",
    "ARIMA",
    "ExponentialSmoothing",
    "AutoCorrelation",
    # Changepoint Detection
    "Changepoint",
    "ChangepointResult",
    "CUSUM",
    "PELT",
    "BayesianChangepoint",
    "segment_statistics",
    # Stability Analysis
    "LyapunovResult",
    "FixedPoint",
    "LyapunovExponents",
    "LinearStability",
    "FixedPointFinder",
    "is_chaotic",
    "stability_classification",
    "find_equilibria",
    # Randomized Linear Algebra
    "TruncatedSVDResult",
    "RandomizedSVD",
    "JohnsonLindenstrauss",
    "RandomizedNystrom",
    "randomized_svd",
    "jl_project",
    # Hidden Markov Models
    "HMMParams",
    "GaussianHMMParams",
    "HMMResult",
    "HiddenMarkovModel",
    "GaussianHMM",
    # Fokker-Planck Equation
    "FPESolution",
    "FokkerPlanckSolver",
    # Bifurcation Analysis
    "BifurcationPoint",
    "BifurcationDiagram",
    "BifurcationAnalyzer",
    "HopfBifurcation",
    "saddle_node_normal_form",
    "pitchfork_normal_form",
    "transcritical_normal_form",
    # Kernel Regression
    "KernelRegressionResult",
    "rbf_kernel",
    "epanechnikov_kernel",
    "tricube_kernel",
    "NadarayaWatson",
    "LocalLinearRegression",
    "KernelDensityEstimation",
    "RKHS",
    # Chaos & Dynamical Systems
    "AttractorResult",
    "StrangeAttractorDetector",
    "PredictabilityAnalyzer",
    "lorenz_system",
    "rossler_system",
    "henon_map",
    "analyze_attractor",
    # Stiff ODE Solvers
    "ODEResult",
    "BDF",
    "ImplicitEuler",
    "TrapezoidalRule",
    # Neural ODEs
    "NeuralODEResult",
    "AdjointResult",
    "CNFResult",
    "NeuralODE",
    "AdjointMethod",
    "ContinuousNormalizing",
    # Poincare Sections
    "SectionResult",
    "ReturnMapResult",
    "PoincareSection",
    "PoincareMap",
    "winding_number",
    # Survival Analysis
    "SurvivalResult",
    "CoxResult",
    "KaplanMeier",
    "NelsonAalen",
    "CoxProportionalHazards",
    "log_rank_test",
    # Mixture Models
    "MixtureResult",
    "GaussianMixture",
    "BayesianGaussianMixture",
    # Robust Regression
    "RegressionResult",
    "HuberRegression",
    "RANSACRegression",
    "TheilSenEstimator",
    # Nonparametric Tests
    "WilcoxonTest",
    "MannWhitneyU",
    "KruskalWallis",
    "FriedmanTest",
    "multiple_testing_correction",
    # Ergodic Theory
    "ErgodicResult",
    "MixingResult",
    "RecurrenceResult",
    "ErgodicAnalyzer",
    "MixingAnalyzer",
    "RecurrenceAnalyzer",
    "is_ergodic",
    # Wasserstein Distances
    "TransportResult",
    "WassersteinDistance",
    "SinkhornDivergence",
    "WassersteinBarycenter",
    "SlicedWasserstein",
    # Bayesian Nonparametrics
    "DPMixtureResult",
    "DirichletProcess",
    "ChineseRestaurantProcess",
    "DPMixture",
    "expected_clusters",
    # Copulas
    "CopulaResult",
    "GaussianCopula",
    "ClaytonCopula",
    "FrankCopula",
    "kendall_tau",
    # ADMM
    "ADMMResult",
    "soft_threshold",
    "ADMM",
    "ConsensusADMM",
    # Spherical Harmonics
    "AssociatedLegendre",
    "SphericalHarmonics",
    # Factor Analysis
    "FactorResult",
    "FactorAnalysis",
    # Discriminant Analysis
    "LDAResult",
    "LDA",
    "QDA",
    # Bootstrap & Resampling
    "Bootstrap",
    "PermutationTest",
    "Jackknife",
    # Streaming Data Structures
    "CountMinSketch",
    "HyperLogLog",
    "ReservoirSampling",
    "ExponentialHistogram",
    # Density Estimation
    "DensityResult",
    "HistogramEstimator",
    "KDEEstimator",
    "WaveletDensity",
    # Spline Regression & VAR
    "SplineResult",
    "VARResult",
    "LOESSRegression",
    "SplineRegression",
    "VARModel",
    "RegressionDiagnostics",
    # Double ML & Synthetic Control
    "DMLResult",
    "SyntheticResult",
    "DoubleML",
    "SyntheticControl",
    # Kolmogorov Complexity
    "ComplexityResult",
    "KolmogorovComplexity",
    "AlgorithmicInformation",
    "BerryParadox",
    # Convex Extensions
    "ConjugateResult",
    "ProxResult",
    "FenchelConjugate",
    "MoreauEnvelope",
    "ProximalOperator",
    # Topological Entropy & KAM
    "KAMResult",
    "TopologicalEntropyResult",
    "LyapunovSpectrumResult",
    "KAMTheory",
    "TopologicalEntropy",
    "LyapunovSpectrum",
    # Stratonovich SDE
    "StratonovichSDE", "StratonovichSDEResult",
    # PDE Solvers
    "WaveResult", "WaveEquationSolver",
    "LaplaceResult", "LaplaceEquationSolver",
    "FEMResult", "FiniteElementMethod",
    # Hamilton-Jacobi
    "HJResult", "HamiltonJacobiSolver",
    # Exterior Calculus
    "WedgeProduct", "HodgeStar", "ExteriorDerivative", "DeRhamCohomology",
    # Exotic Calculus
    "FractionalCalculus", "JetData", "JetBundle",
    "UltrafilterConvergence", "PrincipalUltrafilter",
    # Classical Theorems
    "StoneWeierstrass", "ArzelaAscoli", "WeakTopology",
    "HahnBanach", "HaarMeasure",
    # Harmonic Analysis on Groups & Riemann Surfaces
    "RiemannSurfaceData", "RiemannSurface", "GroupHarmonicAnalysis",

    # Order Statistics & L-Moments
    "OrderStatistics",
    "LMoments",

    # Stable Distributions & Advanced Statistical Methods
    "StableDistribution",
    "VariationalInference",
    "EmpiricalBayes",
    "CCAResult",
    "CanonicalCorrelation",
    "MDS",
    "KernelDensity1D",
    "RankTest",
    "RenyiDP",
    "LocalDP",

    # Multivariate Regression
    "MultivariateResult",
    "MultivariateRegression",
]
