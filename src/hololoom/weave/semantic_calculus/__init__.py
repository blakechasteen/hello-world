"""
Semantic Flow Calculus - Differential Geometry for Language

Mathematical framework treating language as trajectories through semantic space.
Enables interpretable analysis of how meaning changes, with what velocity,
acceleration, and curvature.

Core Innovation:
    Project 384D embeddings -> 16D interpretable semantic dimensions FIRST,
    then apply calculus there. This gives:
    - Speed: 16D vs 384D operations
    - Interpretability: "Warmth +0.3, Formality -0.5"
    - Conservation: Hamiltonian energy preserved

Mathematical Pillars:
    1. Differential Geometry: position, velocity, acceleration, curvature
    2. Semantic Dimensions: 16 interpretable conjugate pairs (Warmth<->Coldness, etc.)
    3. Geometric Integration: structure-preserving Hamiltonian dynamics
    4. Ethical Policy: multi-objective optimization with moral constraints
    5. Integral Geometry: tomographic reconstruction from context slices
    6. Hyperbolic Space: hierarchical concept embeddings (Poincare ball)
    7. System Identification: learn dynamics from data (DE + Linear + Regression)

Quick Start:
    >>> from HoloLoom.semantic_calculus import SemanticFlowCalculus, SemanticSpectrum
    >>> from HoloLoom.embedding.spectral import create_embedder
    >>>
    >>> # Create embedder and calculus engine
    >>> embed_model = create_embedder(sizes=[384])
    >>> embed_fn = lambda word: embed_model.encode([word])[0]
    >>> calculus = SemanticFlowCalculus(embed_fn, dt=1.0)
    >>>
    >>> # Compute trajectory
    >>> words = ["happy", "joyful", "ecstatic", "thrilled"]
    >>> trajectory = calculus.compute_trajectory(words)
    >>> print(f"Curvature: {trajectory.curvature}")
    >>>
    >>> # Project onto interpretable dimensions
    >>> spectrum = SemanticSpectrum()
    >>> spectrum.learn_axes(embed_fn)
    >>> analysis = spectrum.analyze_semantic_forces(trajectory.positions)
    >>> print(f"Dominant dimensions: {analysis['dominant_velocity']}")

Warp Thread Architecture:
    This module is a warp thread - independent and reusable.
    It can be woven into the main pipeline via SEMANTIC_FLOW pattern card
    or used standalone for semantic analysis.

Author: BearL Labs
"""

__version__ = "0.1.0"

# === Core Calculus ===
from .flow_calculus import (
    SemanticState,
    SemanticTrajectory,
    SemanticFlowCalculus,
    SemanticFlowVisualizer,
    analyze_text_flow,
)

# === Semantic Dimensions (The Key Projection!) ===
from .dimensions import (
    SemanticDimension,
    SemanticSpectrum,
    STANDARD_DIMENSIONS,
    EXTENDED_244_DIMENSIONS,
    visualize_semantic_spectrum,
    print_spectrum_summary,
)

# === Geometric Integration ===
from .integrator import (
    GeometricIntegrator,
    MultiScaleGeometricFlow,
    visualize_geometric_flow,
    compute_semantic_force_field,
)
# Note: integrator.SemanticState is not exported (internal, conflicts with flow_calculus.SemanticState)

# === Ethical Policy ===
from .ethics import (
    EthicalObjective,
    EthicalSemanticPolicy,
    COMPASSIONATE_COMMUNICATION,
    SCIENTIFIC_DISCOURSE,
    THERAPEUTIC_DIALOGUE,
    visualize_ethical_landscape,
)

# === Integral Geometry ===
from .integral_geometry import (
    RadonTransform,
    InverseRadonTransform,
    CroftonFormula,
    SemanticTomography,
    visualize_tomographic_reconstruction,
)

# === Hyperbolic Semantics ===
from .hyperbolic import (
    HyperbolicPoint,
    PoincareGeometry,
    HyperbolicSemanticSpace,
    ComplexSemanticFlow,
    SemanticSymmetryGroup,
    visualize_hyperbolic_hierarchy,
)

# === System Identification ===
from .system_id import (
    LearnedSemanticSystem,
    SemanticSystemIdentification,
    visualize_system_identification,
    demonstrate_system_identification,
)

# === Performance Utilities ===
from .performance import (
    EmbeddingCache,
    ProjectionCache,
    timer,
    HAS_NUMBA,
)

# === Integration Layer (Clean Interface) ===
# Now organized into focused modules
from .config import SemanticCalculusConfig
from .analyzer import SemanticAnalyzer, create_semantic_analyzer, get_cache_stats
from .adapter import create_semantic_thread, quick_analysis, format_semantic_summary, extract_trajectory_metrics

# Backward compatibility: Keep old integration.py imports working
try:
    from .integration import (
        SemanticCalculusConfig as _LegacyConfig,
        SemanticAnalyzer as _LegacyAnalyzer,
        create_semantic_analyzer as _legacy_create,
    )
except ImportError:
    # integration.py may not exist anymore, which is fine
    pass

# === Public API ===
__all__ = [
    # Core calculus
    "SemanticState",
    "SemanticTrajectory",
    "SemanticFlowCalculus",
    "SemanticFlowVisualizer",
    "analyze_text_flow",
    # Dimensions
    "SemanticDimension",
    "SemanticSpectrum",
    "STANDARD_DIMENSIONS",
    "EXTENDED_244_DIMENSIONS",
    "visualize_semantic_spectrum",
    "print_spectrum_summary",
    # Integration
    "GeometricIntegrator",
    "MultiScaleGeometricFlow",
    "visualize_geometric_flow",
    "compute_semantic_force_field",
    # Ethics
    "EthicalObjective",
    "EthicalSemanticPolicy",
    "COMPASSIONATE_COMMUNICATION",
    "SCIENTIFIC_DISCOURSE",
    "THERAPEUTIC_DIALOGUE",
    "visualize_ethical_landscape",
    # Integral geometry
    "RadonTransform",
    "InverseRadonTransform",
    "CroftonFormula",
    "SemanticTomography",
    "visualize_tomographic_reconstruction",
    # Hyperbolic
    "HyperbolicPoint",
    "PoincareGeometry",
    "HyperbolicSemanticSpace",
    "ComplexSemanticFlow",
    "SemanticSymmetryGroup",
    "visualize_hyperbolic_hierarchy",
    # System ID
    "LearnedSemanticSystem",
    "SemanticSystemIdentification",
    "visualize_system_identification",
    "demonstrate_system_identification",
    # Performance
    "EmbeddingCache",
    "ProjectionCache",
    "timer",
    "HAS_NUMBA",
    # Integration Layer
    "SemanticCalculusConfig",
    "SemanticAnalyzer",
    "create_semantic_analyzer",
    "create_semantic_thread",
    "quick_analysis",
    "get_cache_stats",
    "format_semantic_summary",
    "extract_trajectory_metrics",
]


# === Clean Imports (Recommended) ===
# For new code, use the organized structure:
#
#   from HoloLoom.semantic_calculus import create_semantic_analyzer
#   from HoloLoom.semantic_calculus.math import SemanticFlow, SemanticSpectrum
#   from HoloLoom.semantic_calculus.config import SemanticCalculusConfig
#
# Legacy imports still work for backward compatibility:
#
#   from HoloLoom.semantic_calculus import SemanticFlowCalculus  # Still works!
#


# === Convenience Functions ===

def create_semantic_analyzer(embed_fn, dt=1.0, mass=1.0):
    """
    Create a complete semantic analysis pipeline

    Args:
        embed_fn: Function word -> embedding vector
        dt: Time step for calculus
        mass: Semantic mass for dynamics

    Returns:
        Dictionary with {calculus, spectrum, integrator, policy}
    """
    # Create calculus engine
    calculus = SemanticFlowCalculus(embed_fn, dt=dt, mass=mass)

    # Create spectrum analyzer
    spectrum = SemanticSpectrum()
    spectrum.learn_axes(embed_fn)

    # Create geometric integrator
    # Build projection matrix from learned dimensions
    import numpy as np
    P = np.array([dim.axis for dim in spectrum.dimensions])  # (16, 384)
    integrator = GeometricIntegrator(P, mass=mass)

    # Create ethical policy (default: compassionate)
    dim_names = [dim.name for dim in spectrum.dimensions]
    policy = EthicalSemanticPolicy(COMPASSIONATE_COMMUNICATION, dim_names)

    return {
        'calculus': calculus,
        'spectrum': spectrum,
        'integrator': integrator,
        'policy': policy,
    }


def analyze_conversation(words, embed_fn, dt=1.0):
    """
    Quick one-shot analysis of a conversation

    Args:
        words: List of words/tokens
        embed_fn: Embedding function
        dt: Time step

    Returns:
        Dictionary with complete analysis
    """
    # Create analyzer
    analyzer = create_semantic_analyzer(embed_fn, dt=dt)

    # Compute trajectory
    trajectory = analyzer['calculus'].compute_trajectory(words)

    # Project onto dimensions
    analysis = analyzer['spectrum'].analyze_semantic_forces(
        trajectory.positions, dt=dt
    )

    # Evaluate ethics
    ethical_analysis = analyzer['policy'].analyze_conversation_ethics(
        analyzer['integrator'].project_to_semantic(trajectory.positions.T).T
    )

    return {
        'trajectory': trajectory,
        'semantic_forces': analysis,
        'ethics': ethical_analysis,
        'analyzer': analyzer,
    }
