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

Complete 10-module analysis suite for rigorous mathematics in AI.
"""

from .real_analysis import (
    MetricSpace,
    SequenceAnalyzer,
    ContinuityChecker,
    Differentiator,
    RiemannIntegrator
)

from .complex_analysis import (
    ComplexFunction,
    ContourIntegrator,
    ResidueCalculator,
    ConformalMapper,
    SeriesExpansion,
    AnalyticContinuation
)

from .functional_analysis import (
    NormedSpace,
    HilbertSpace,
    BoundedOperator,
    SpectralAnalyzer,
    SobolevSpace,
    CompactOperator
)

from .measure_theory import (
    SigmaAlgebra,
    Measure,
    LebesgueMeasure,
    MeasurableFunction,
    LebesgueIntegrator,
    ConvergenceTheorems
)

from .fourier_harmonic import (
    FourierTransform,
    FourierSeries,
    WaveletTransform,
    TimeFrequencyAnalysis
)

from .stochastic_calculus import (
    BrownianMotion,
    MartingaleAnalyzer,
    ItoIntegrator,
    ItosLemma,
    StochasticDifferentialEquation,
    SDEResult
)

from .advanced_topics import (
    WaveFrontSet,
    PseudodifferentialOperator,
    Hyperreal,
    NonstandardAnalysis,
    PAdicNumber,
    HenselsLemma
)

from .numerical_analysis import (
    RootFinder,
    NumericalLinearAlgebra,
    ODESolver,
    ODESolution,
    Interpolation,
    NumericalOptimization
)

from .probability_theory import (
    ProbabilitySpace,
    RandomVariable,
    CommonDistributions,
    LimitTheorems,
    MaximumLikelihoodEstimation,
    BayesianInference,
    HypothesisTesting,
    MarkovChain
)

from .distribution_theory import (
    SchwartzFunction,
    Distribution,
    StandardDistributions,
    DistributionFourier,
    GreenFunction,
    WeakDerivative
)

__all__ = [
    # Real Analysis
    "MetricSpace",
    "SequenceAnalyzer",
    "ContinuityChecker",
    "Differentiator",
    "RiemannIntegrator",
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
]
