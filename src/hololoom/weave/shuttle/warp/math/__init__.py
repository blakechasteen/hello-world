"""
HoloLoom Mathematical Modules
==============================

Advanced mathematics for rigorous analysis and computation.

Modules:
- analysis: Real, complex, measure, functional analysis
- algebra: (coming soon) Abstract algebra, Lie algebras
- geometry: (coming soon) Differential geometry, algebraic geometry
- probability: (coming soon) Measure-theoretic probability
"""

# Analysis modules
try:
    from .analysis.real_analysis import (
        MetricSpace,
        SequenceAnalyzer,
        ContinuityChecker,
        Differentiator,
        RiemannIntegrator
    )
    HAS_REAL_ANALYSIS = True
except ImportError:
    HAS_REAL_ANALYSIS = False

try:
    from .analysis.complex_analysis import (
        ComplexFunction,
        ContourIntegrator,
        ResidueCalculator,
        ConformalMapper,
        SeriesExpansion,
        AnalyticContinuation
    )
    HAS_COMPLEX_ANALYSIS = True
except ImportError:
    HAS_COMPLEX_ANALYSIS = False

__all__ = []

if HAS_REAL_ANALYSIS:
    __all__.extend([
        "MetricSpace",
        "SequenceAnalyzer",
        "ContinuityChecker",
        "Differentiator",
        "RiemannIntegrator"
    ])

if HAS_COMPLEX_ANALYSIS:
    __all__.extend([
        "ComplexFunction",
        "ContourIntegrator",
        "ResidueCalculator",
        "ConformalMapper",
        "SeriesExpansion",
        "AnalyticContinuation"
    ])
