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
    >>> from hololoom.semantic_calculus.math import SemanticFlow, SemanticSpectrum
    >>> flow = SemanticFlow(embed_fn)
    >>> trajectory = flow.compute_trajectory(words)
    >>> spectrum = SemanticSpectrum()
    >>> analysis = spectrum.analyze_semantic_forces(trajectory.positions)
"""

# Core trajectory calculus (from flow_calculus.py)
# Semantic dimension projection (from dimensions.py)
from ..dimensions import (
    STANDARD_DIMENSIONS,
    SemanticDimension,
    SemanticSpectrum,
    print_spectrum_summary,
    visualize_semantic_spectrum,
)

# Ethical optimization (from ethics.py)
from ..ethics import (
    COMPASSIONATE_COMMUNICATION,
    SCIENTIFIC_DISCOURSE,
    THERAPEUTIC_DIALOGUE,
    EthicalObjective,
    visualize_ethical_landscape,
)
from ..ethics import (
    EthicalSemanticPolicy as EthicalPolicy,  # Shorter name
)
from ..flow_calculus import (
    SemanticFlowCalculus as SemanticFlow,  # Cleaner name
)
from ..flow_calculus import (
    SemanticFlowVisualizer,
    SemanticState,
    SemanticTrajectory,
    analyze_text_flow,
)

# Hyperbolic geometry (from hyperbolic.py)
from ..hyperbolic import (
    ComplexSemanticFlow,
    HyperbolicPoint,
    HyperbolicSemanticSpace,
    PoincareGeometry,
    SemanticSymmetryGroup,
    visualize_hyperbolic_hierarchy,
)

# Integral geometry / tomography (from integral_geometry.py)
from ..integral_geometry import (
    CroftonFormula,
    InverseRadonTransform,
    RadonTransform,
    SemanticTomography,
    visualize_tomographic_reconstruction,
)

# Hamiltonian dynamics (from integrator.py)
from ..integrator import (
    GeometricIntegrator as HamiltonianDynamics,  # More descriptive name
)
from ..integrator import (
    MultiScaleGeometricFlow as MultiScaleFlow,
)
from ..integrator import (
    compute_semantic_force_field,
    visualize_geometric_flow,
)

# System identification (from system_id.py)
from ..system_id import (
    LearnedSemanticSystem,
    SemanticSystemIdentification,
    demonstrate_system_identification,
    visualize_system_identification,
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
