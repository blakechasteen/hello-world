"""
HoloLoom Integration Adapters
==============================
Helper functions for integrating semantic calculus with HoloLoom components.
"""

from typing import Dict, Any, Callable

from .analyzer import SemanticAnalyzer
from .config import SemanticCalculusConfig


def create_semantic_thread(
    analyzer: SemanticAnalyzer,
    text: str,
    weight: float = 1.0
) -> Dict[str, Any]:
    """
    Create a semantic flow thread for ResonanceShed.

    This adapter extracts features and formats them for DotPlasma integration.

    Args:
        analyzer: SemanticAnalyzer instance
        text: Input text to analyze
        weight: Thread weight (default 1.0)

    Returns:
        Dictionary formatted for FeatureThread

    Example:
        >>> analyzer = create_semantic_analyzer(embed_fn)
        >>> thread = create_semantic_thread(analyzer, "Your text...")
        >>> # Pass to ResonanceShed
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


def quick_analysis(text: str, embed_fn: Callable) -> Dict[str, Any]:
    """
    Quick one-shot semantic analysis.

    Uses fast configuration for speed.

    Args:
        text: Text to analyze
        embed_fn: Embedding function

    Returns:
        Analysis results with trajectory and semantic forces

    Example:
        >>> from HoloLoom.embedding.spectral import create_embedder
        >>> embedder = create_embedder(sizes=[384])
        >>> embed_fn = lambda words: embedder.encode(words)
        >>> result = quick_analysis("Your text...", embed_fn)
        >>> print(result['trajectory'].states[0].speed)
    """
    from .analyzer import create_semantic_analyzer

    analyzer = create_semantic_analyzer(
        embed_fn,
        config=SemanticCalculusConfig.fast()
    )
    return analyzer.analyze_text(text)


def extract_trajectory_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract key trajectory metrics from analysis result.

    Args:
        result: Result from analyzer.analyze_text()

    Returns:
        Dictionary with scalar metrics

    Example:
        >>> result = analyzer.analyze_text("Your text...")
        >>> metrics = extract_trajectory_metrics(result)
        >>> print(f"Speed: {metrics['avg_speed']:.3f}")
    """
    trajectory = result.get('trajectory')
    if not trajectory:
        return {}

    states = trajectory.states

    return {
        'n_words': len(states),
        'total_distance': float(trajectory.total_distance()),
        'avg_speed': float(sum(s.speed for s in states) / len(states)) if states else 0.0,
        'max_speed': float(max(s.speed for s in states)) if states else 0.0,
        'avg_acceleration': float(sum(s.acceleration_magnitude for s in states) / len(states)) if states else 0.0,
        'avg_curvature': float(sum(trajectory.curvature(i) for i in range(len(states))) / len(states)) if states else 0.0,
    }


def format_semantic_summary(result: Dict[str, Any]) -> str:
    """
    Format analysis result as human-readable summary.

    Args:
        result: Result from analyzer.analyze_text()

    Returns:
        Formatted summary string

    Example:
        >>> result = analyzer.analyze_text("Your text...")
        >>> print(format_semantic_summary(result))
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SEMANTIC ANALYSIS SUMMARY")
    lines.append("=" * 60)

    # Trajectory metrics
    metrics = extract_trajectory_metrics(result)
    if metrics:
        lines.append(f"\nTrajectory:")
        lines.append(f"  Words: {metrics['n_words']}")
        lines.append(f"  Distance: {metrics['total_distance']:.3f}")
        lines.append(f"  Avg Speed: {metrics['avg_speed']:.3f}")
        lines.append(f"  Avg Curvature: {metrics['avg_curvature']:.3f}")

    # Semantic forces
    forces = result.get('semantic_forces', {})
    if 'dominant_velocity' in forces:
        lines.append(f"\nDominant Dimensions:")
        for dim, score in forces['dominant_velocity'][:3]:
            lines.append(f"  {dim}: {score:.3f}")

    # Ethics
    ethics = result.get('ethics', {})
    if ethics:
        lines.append(f"\nEthics:")
        lines.append(f"  Virtue Score: {ethics.get('total_virtue', 0):.3f}")
        if ethics.get('manipulation_patterns'):
            lines.append(f"  ⚠️  Manipulation detected")

    lines.append("=" * 60)
    return "\n".join(lines)
