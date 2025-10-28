"""
Pure Mathematical Operations for Semantic Calculus
===================================================

This module contains pure mathematical implementations independent of HoloLoom.
All operations are on embedding manifolds with geometric structure.

Can be used standalone or as part of the HoloLoom weaving system.

Modules:
- flow: Trajectory computation, velocity, acceleration, curvature
- spectrum: 16D semantic dimension projection and analysis
- dynamics: Hamiltonian mechanics and geometric integration
- optimization: Multi-objective ethical optimization

Example:
    >>> from HoloLoom.semantic_calculus.math import SemanticFlow, SemanticSpectrum
    >>> flow = SemanticFlow(embed_fn)
    >>> trajectory = flow.compute_trajectory(words)
    >>> spectrum = SemanticSpectrum()
    >>> analysis = spectrum.analyze_semantic_forces(trajectory.positions)
"""

# Core trajectory calculus (from flow_calculus.py)
from ..flow_calculus import (
    SemanticState,
    SemanticTrajectory,
    SemanticFlowCalculus as SemanticFlow,  # Cleaner name
    SemanticFlowVisualizer,
    analyze_text_flow,
)

# Semantic dimension projection (from dimensions.py)
from ..dimensions import (
    SemanticDimension,
    SemanticSpectrum,
    STANDARD_DIMENSIONS,
    visualize_semantic_spectrum,
    print_spectrum_summary,
)

# Hamiltonian dynamics (from integrator.py)
from ..integrator import (
    GeometricIntegrator as HamiltonianDynamics,  # More descriptive name
    MultiScaleGeometricFlow as MultiScaleFlow,
    visualize_geometric_flow,
    compute_semantic_force_field,
)

# Ethical optimization (from ethics.py)
from ..ethics import (
    EthicalObjective,
    EthicalSemanticPolicy as EthicalPolicy,  # Shorter name
    COMPASSIONATE_COMMUNICATION,
    SCIENTIFIC_DISCOURSE,
    THERAPEUTIC_DIALOGUE,
    visualize_ethical_landscape,
)

# Hyperbolic geometry (from hyperbolic.py)
from ..hyperbolic import (
    HyperbolicPoint,
    PoincareGeometry,
    HyperbolicSemanticSpace,
    ComplexSemanticFlow,
    SemanticSymmetryGroup,
    visualize_hyperbolic_hierarchy,
)

# Integral geometry / tomography (from integral_geometry.py)
from ..integral_geometry import (
    RadonTransform,
    InverseRadonTransform,
    CroftonFormula,
    SemanticTomography,
    visualize_tomographic_reconstruction,
)

# System identification (from system_id.py)
from ..system_id import (
    LearnedSemanticSystem,
    SemanticSystemIdentification,
    visualize_system_identification,
    demonstrate_system_identification,
)

__all__ = [
    # Core flow
    "SemanticState",
    "SemanticTrajectory",
    "SemanticFlow",
    "SemanticFlowVisualizer",
    "analyze_text_flow",

    # Spectrum
    "SemanticDimension",
    "SemanticSpectrum",
    "STANDARD_DIMENSIONS",
    "visualize_semantic_spectrum",
    "print_spectrum_summary",

    # Dynamics
    "HamiltonianDynamics",
    "MultiScaleFlow",
    "visualize_geometric_flow",
    "compute_semantic_force_field",

    # Ethics
    "EthicalObjective",
    "EthicalPolicy",
    "COMPASSIONATE_COMMUNICATION",
    "SCIENTIFIC_DISCOURSE",
    "THERAPEUTIC_DIALOGUE",
    "visualize_ethical_landscape",

    # Hyperbolic
    "HyperbolicPoint",
    "PoincareGeometry",
    "HyperbolicSemanticSpace",
    "ComplexSemanticFlow",
    "SemanticSymmetryGroup",
    "visualize_hyperbolic_hierarchy",

    # Tomography
    "RadonTransform",
    "InverseRadonTransform",
    "CroftonFormula",
    "SemanticTomography",
    "visualize_tomographic_reconstruction",

    # System ID
    "LearnedSemanticSystem",
    "SemanticSystemIdentification",
    "visualize_system_identification",
    "demonstrate_system_identification",
]

__version__ = "1.0.0"
