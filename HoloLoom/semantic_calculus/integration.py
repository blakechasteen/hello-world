"""
Semantic Calculus Integration Layer
====================================
Clean integration between semantic calculus and HoloLoom weaving pipeline.

This module provides:
1. Factory functions for creating semantic analyzers
2. Configuration helpers
3. Feature extraction adapters
4. DotPlasma integration utilities

Architecture:
    This is the "adapter layer" between the semantic calculus warp thread
    and the main weaving shuttle. It translates between:
    - Pattern card specs -> Semantic analyzer configuration
    - Trajectory data -> DotPlasma features
    - Embedder functions -> Calculus-compatible interface
"""

from typing import Dict, Any, Optional, Callable, List
import logging
import numpy as np

from . import (
    SemanticFlowCalculus,
    SemanticSpectrum,
    EthicalSemanticPolicy,
    COMPASSIONATE_COMMUNICATION,
    SCIENTIFIC_DISCOURSE,
    THERAPEUTIC_DIALOGUE,
    GeometricIntegrator,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class SemanticCalculusConfig:
    """Configuration for semantic calculus integration."""

    def __init__(
        self,
        enable_cache: bool = True,
        cache_size: int = 10000,
        dimensions: int = 16,
        dt: float = 1.0,
        mass: float = 1.0,
        ethical_framework: str = "compassionate",
        compute_trajectory: bool = True,
        compute_ethics: bool = True,
    ):
        """
        Initialize semantic calculus configuration.

        Args:
            enable_cache: Enable embedding cache for performance
            cache_size: Maximum number of cached embeddings
            dimensions: Number of semantic dimensions (default 16)
            dt: Time step for calculus operations
            mass: Semantic mass parameter for dynamics
            ethical_framework: "compassionate", "scientific", or "therapeutic"
            compute_trajectory: Compute velocity/acceleration/curvature
            compute_ethics: Run ethical analysis
        """
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.dimensions = dimensions
        self.dt = dt
        self.mass = mass
        self.ethical_framework = ethical_framework
        self.compute_trajectory = compute_trajectory
        self.compute_ethics = compute_ethics

    @classmethod
    def from_pattern_spec(cls, pattern_spec) -> 'SemanticCalculusConfig':
        """Create config from pattern card specification."""
        return cls(
            enable_cache=True,  # Always enable cache for performance
            cache_size=10000,
            dimensions=getattr(pattern_spec, 'semantic_dimensions', 16),
            dt=1.0,
            mass=1.0,
            ethical_framework="compassionate",
            compute_trajectory=getattr(pattern_spec, 'semantic_trajectory', True),
            compute_ethics=getattr(pattern_spec, 'semantic_ethics', True),
        )

    @classmethod
    def fast(cls) -> 'SemanticCalculusConfig':
        """Fast configuration (minimal features)."""
        return cls(
            enable_cache=True,
            cache_size=5000,
            dimensions=8,  # Fewer dimensions
            compute_trajectory=True,
            compute_ethics=False,  # Skip ethics for speed
        )

    @classmethod
    def balanced(cls) -> 'SemanticCalculusConfig':
        """Balanced configuration (default)."""
        return cls(
            enable_cache=True,
            cache_size=10000,
            dimensions=16,
            compute_trajectory=True,
            compute_ethics=True,
        )

    @classmethod
    def comprehensive(cls) -> 'SemanticCalculusConfig':
        """Comprehensive configuration (all features)."""
        return cls(
            enable_cache=True,
            cache_size=20000,
            dimensions=32,  # More dimensions for detail
            compute_trajectory=True,
            compute_ethics=True,
        )


# ============================================================================
# Factory Functions
# ============================================================================

class SemanticAnalyzer:
    """
    Unified semantic analysis pipeline.

    Combines calculus, spectrum, integrator, and policy into single interface.
    This is the main entry point for semantic analysis in HoloLoom.
    """

    def __init__(
        self,
        calculus: SemanticFlowCalculus,
        spectrum: SemanticSpectrum,
        integrator: Optional[GeometricIntegrator] = None,
        policy: Optional[EthicalSemanticPolicy] = None,
        config: Optional[SemanticCalculusConfig] = None,
    ):
        """
        Initialize semantic analyzer.

        Args:
            calculus: Semantic flow calculus engine
            spectrum: Semantic spectrum analyzer
            integrator: Optional geometric integrator
            policy: Optional ethical policy
            config: Configuration object
        """
        self.calculus = calculus
        self.spectrum = spectrum
        self.integrator = integrator
        self.policy = policy
        self.config = config or SemanticCalculusConfig.balanced()

        logger.info("SemanticAnalyzer initialized")

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Complete semantic analysis of text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with trajectory, dimensions, and ethics
        """
        words = text.split()

        # Compute trajectory
        trajectory = self.calculus.compute_trajectory(words)

        # Project onto semantic dimensions
        semantic_forces = self.spectrum.analyze_semantic_forces(
            trajectory.positions,
            dt=self.config.dt
        )

        result = {
            'trajectory': trajectory,
            'semantic_forces': semantic_forces,
            'n_words': len(words),
        }

        # Optional: Ethical analysis
        if self.config.compute_ethics and self.policy:
            q_semantic = self.integrator.project_to_semantic(trajectory.positions.T).T
            ethical_analysis = self.policy.analyze_conversation_ethics(q_semantic)
            result['ethics'] = ethical_analysis

        return result

    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract features for DotPlasma integration.

        This is the adapter method used by ResonanceShed.

        Args:
            text: Input text

        Returns:
            Dictionary of features compatible with DotPlasma
        """
        words = text.split()
        trajectory = self.calculus.compute_trajectory(words)

        # Extract core metrics
        features = {
            'n_states': len(trajectory.states),
            'total_distance': float(trajectory.total_distance()),
        }

        # Trajectory features (if enabled)
        if self.config.compute_trajectory:
            speeds = [s.speed for s in trajectory.states]
            accels = [s.acceleration_magnitude for s in trajectory.states]
            curvatures = [trajectory.curvature(i) for i in range(len(trajectory.states))]

            features.update({
                'avg_velocity': float(np.mean(speeds)) if speeds else 0.0,
                'max_velocity': float(np.max(speeds)) if speeds else 0.0,
                'avg_acceleration': float(np.mean(accels)) if accels else 0.0,
                'curvature': curvatures,
                'avg_curvature': float(np.mean(curvatures)) if curvatures else 0.0,
                'trajectory': trajectory,  # Store full trajectory for later use
            })

        # Semantic dimension analysis
        semantic_forces = self.spectrum.analyze_semantic_forces(
            trajectory.positions,
            dt=self.config.dt
        )
        features['semantic_forces'] = semantic_forces

        # Dominant dimensions
        if 'dominant_velocity' in semantic_forces:
            features['dominant_dimensions'] = semantic_forces['dominant_velocity'][:5]

        # Ethical analysis (if enabled)
        if self.config.compute_ethics and self.policy and self.integrator:
            q_semantic = self.integrator.project_to_semantic(trajectory.positions.T).T
            ethical_analysis = self.policy.analyze_conversation_ethics(q_semantic)
            features['ethics'] = {
                'virtue_score': ethical_analysis.get('total_virtue', 0.0),
                'manipulation_detected': bool(ethical_analysis.get('manipulation_patterns')),
            }

        return features


def create_semantic_analyzer(
    embed_fn: Callable,
    config: Optional[SemanticCalculusConfig] = None,
) -> SemanticAnalyzer:
    """
    Factory function to create complete semantic analyzer.

    This is the main entry point for creating semantic analysis pipelines.

    Args:
        embed_fn: Embedding function (word or list -> embeddings)
        config: Optional configuration (uses balanced() if None)

    Returns:
        Configured SemanticAnalyzer instance

    Example:
        >>> from HoloLoom.embedding.spectral import create_embedder
        >>> embed_model = create_embedder(sizes=[384])
        >>> embed_fn = lambda words: embed_model.encode(words)
        >>> analyzer = create_semantic_analyzer(embed_fn)
        >>> result = analyzer.analyze_text("Hello world")
    """
    config = config or SemanticCalculusConfig.balanced()

    logger.info(f"Creating semantic analyzer: {config.dimensions}D, "
                f"cache={'enabled' if config.enable_cache else 'disabled'}")

    # Create calculus engine
    calculus = SemanticFlowCalculus(
        embed_fn,
        dt=config.dt,
        enable_cache=config.enable_cache,
        cache_size=config.cache_size,
    )

    # Create spectrum analyzer
    spectrum = SemanticSpectrum(dim_reduction='pca')
    spectrum.learn_axes(embed_fn, n_dims=config.dimensions)

    # Create geometric integrator
    integrator = None
    if config.compute_ethics:
        P = np.array([dim.axis for dim in spectrum.dimensions])  # (n_dims, 384)
        integrator = GeometricIntegrator(P, mass=config.mass)

    # Create ethical policy
    policy = None
    if config.compute_ethics:
        frameworks = {
            'compassionate': COMPASSIONATE_COMMUNICATION,
            'scientific': SCIENTIFIC_DISCOURSE,
            'therapeutic': THERAPEUTIC_DIALOGUE,
        }
        framework = frameworks.get(config.ethical_framework, COMPASSIONATE_COMMUNICATION)
        dim_names = [dim.name for dim in spectrum.dimensions]
        policy = EthicalSemanticPolicy(framework, dim_names)

    return SemanticAnalyzer(
        calculus=calculus,
        spectrum=spectrum,
        integrator=integrator,
        policy=policy,
        config=config,
    )


# ============================================================================
# ResonanceShed Adapter
# ============================================================================

def create_semantic_thread(analyzer: SemanticAnalyzer, text: str, weight: float = 1.0) -> Dict[str, Any]:
    """
    Create a semantic flow thread for ResonanceShed.

    This adapter extracts features and formats them for DotPlasma integration.

    Args:
        analyzer: SemanticAnalyzer instance
        text: Input text to analyze
        weight: Thread weight (default 1.0)

    Returns:
        Dictionary formatted for FeatureThread
    """
    features = analyzer.extract_features(text)

    return {
        'name': 'semantic_flow',
        'features': features,
        'weight': weight,
        'metadata': {
            'n_words': features.get('n_states', 0),
            'dimensions': analyzer.config.dimensions,
            'cache_enabled': analyzer.config.enable_cache,
            'ethics_enabled': analyzer.config.compute_ethics,
        }
    }


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_analysis(text: str, embed_fn: Callable) -> Dict[str, Any]:
    """
    Quick one-shot semantic analysis.

    Args:
        text: Text to analyze
        embed_fn: Embedding function

    Returns:
        Analysis results
    """
    analyzer = create_semantic_analyzer(embed_fn, config=SemanticCalculusConfig.fast())
    return analyzer.analyze_text(text)


def get_cache_stats(analyzer: SemanticAnalyzer) -> Dict[str, Any]:
    """
    Get cache statistics from analyzer.

    Args:
        analyzer: SemanticAnalyzer instance

    Returns:
        Cache statistics
    """
    if analyzer.calculus._cache:
        return analyzer.calculus._cache.get_stats()
    return {'cache_enabled': False}
