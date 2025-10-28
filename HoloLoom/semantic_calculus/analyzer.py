"""
Semantic Analyzer - Unified Analysis Pipeline
==============================================
Main interface for semantic calculus analysis.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Callable

from .config import SemanticCalculusConfig
from .flow_calculus import SemanticFlowCalculus
from .dimensions import (
    SemanticSpectrum,
    STANDARD_DIMENSIONS,
    EXTENDED_244_DIMENSIONS,
)
from .dimension_selector import (
    SmartDimensionSelector,
    SelectionStrategy,
    create_fused_36d_selection,
)
from .integrator import GeometricIntegrator
from .ethics import (
    EthicalSemanticPolicy,
    COMPASSIONATE_COMMUNICATION,
    SCIENTIFIC_DISCOURSE,
    THERAPEUTIC_DIALOGUE,
)

logger = logging.getLogger(__name__)


class SemanticAnalyzer:
    """
    Unified semantic analysis pipeline.

    Combines calculus, spectrum, integrator, and policy into single interface.
    This is the main entry point for semantic analysis in HoloLoom.

    Example:
        >>> analyzer = create_semantic_analyzer(embed_fn)
        >>> result = analyzer.analyze_text("Your text here...")
        >>> print(result['trajectory'].states[0].speed)
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
        if self.config.compute_ethics and self.policy and self.integrator:
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

    # Select dimension set based on config
    if config.dimensions == 244:
        dimension_set = EXTENDED_244_DIMENSIONS
        logger.info("Using EXTENDED_244_DIMENSIONS for deep narrative research")
    elif config.dimensions == 36:
        # Smart selection for FUSED mode
        strategy_name = config.selection_strategy or 'hybrid'
        domain_name = config.domain or 'narrative'

        # Map string to enum
        strategy_map = {
            'balanced': SelectionStrategy.BALANCED,
            'narrative': SelectionStrategy.NARRATIVE,
            'dialogue': SelectionStrategy.DIALOGUE,
            'hybrid': SelectionStrategy.HYBRID,
            'discriminative': SelectionStrategy.DISCRIMINATIVE,
        }
        strategy = strategy_map.get(strategy_name.lower(), SelectionStrategy.HYBRID)

        logger.info(f"Using SMART SELECTION for FUSED mode (36D from 244D)")
        logger.info(f"  Strategy: {strategy_name.upper()}, Domain: {domain_name}")

        selector = SmartDimensionSelector()
        dimension_set = selector.select(
            n_dimensions=36,
            strategy=strategy,
            embed_fn=embed_fn,
            domain=domain_name
        )
    elif config.dimensions == 16:
        dimension_set = STANDARD_DIMENSIONS
        logger.info("Using STANDARD_DIMENSIONS (16D)")
    else:
        # Use first N dimensions from standard set
        dimension_set = STANDARD_DIMENSIONS[:config.dimensions]
        logger.info(f"Using first {config.dimensions} dimensions from STANDARD_DIMENSIONS")

    # Create spectrum analyzer
    spectrum = SemanticSpectrum(dimensions=dimension_set)
    spectrum.learn_axes(embed_fn)

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


def get_cache_stats(analyzer: SemanticAnalyzer) -> Dict[str, Any]:
    """
    Get cache statistics from analyzer.

    Args:
        analyzer: SemanticAnalyzer instance

    Returns:
        Cache statistics dictionary
    """
    if analyzer.calculus._cache:
        return analyzer.calculus._cache.get_stats()
    return {'cache_enabled': False}
