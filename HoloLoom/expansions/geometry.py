"""
Geometry Expansion Bundle - Riemannian Embeddings
=================================================

Non-Euclidean geometry for embeddings: hyperbolic (hierarchies),
spherical (clusters), and mixed-curvature product manifolds.

Install: pip install hololoom[geometry]

Usage:
    from HoloLoom.config import Config
    from HoloLoom.expansions.geometry import GeometryConfig

    config = Config.research()
    config.load_expansion(GeometryConfig(use_riemannian=True))

Date: December 2025
"""

from dataclasses import dataclass
from typing import Any, Dict

try:
    from HoloLoom.config import ExpansionBundle
except ImportError:
    # Fallback for testing
    class ExpansionBundle:
        def get_settings(self) -> Dict[str, Any]:
            raise NotImplementedError


@dataclass
class GeometryConfig(ExpansionBundle):
    """
    Geometry Expansion Bundle: Riemannian Embeddings.

    Non-Euclidean manifold structure for embeddings:

    - Hyperbolic (K < 0): Better for hierarchical concepts (taxonomies, trees)
      - Exponentially more space at edges
      - Natural for parent-child relationships

    - Spherical (K > 0): Better for clustered concepts (topics, categories)
      - Bounded space, natural clustering
      - Good for circular/periodic patterns

    - Euclidean (K = 0): Traditional linear features
      - Baseline for comparison
      - Good for general-purpose

    Product manifolds combine all three for different aspects of concepts.
    """

    # =========================================================================
    # RIEMANNIAN EMBEDDING SETTINGS (6 fields)
    # =========================================================================

    use_riemannian: bool = False
    """Enable Riemannian manifold structure for embeddings."""

    riemannian_hyperbolic_dim: int = 256
    """Dimension for hierarchical concepts (negative curvature K < 0)."""

    riemannian_spherical_dim: int = 256
    """Dimension for clustered concepts (positive curvature K > 0)."""

    riemannian_euclidean_dim: int = 256
    """Dimension for linear features (zero curvature K = 0)."""

    riemannian_hyperbolic_curvature: float = -1.0
    """
    Hyperbolic curvature (negative).
    - Lower (more negative) = more room at edges (deeper trees)
    - -0.1 to -10.0 typical range
    """

    riemannian_spherical_curvature: float = 1.0
    """
    Spherical curvature (positive).
    - Higher = tighter clusters
    - 0.1 to 10.0 typical range
    """

    def get_settings(self) -> Dict[str, Any]:
        """Return dictionary of settings to merge into config."""
        return {
            "use_riemannian": self.use_riemannian,
            "riemannian_hyperbolic_dim": self.riemannian_hyperbolic_dim,
            "riemannian_spherical_dim": self.riemannian_spherical_dim,
            "riemannian_euclidean_dim": self.riemannian_euclidean_dim,
            "riemannian_hyperbolic_curvature": self.riemannian_hyperbolic_curvature,
            "riemannian_spherical_curvature": self.riemannian_spherical_curvature,
        }

    @property
    def total_dim(self) -> int:
        """Total embedding dimension (sum of all components)."""
        return (
            self.riemannian_hyperbolic_dim +
            self.riemannian_spherical_dim +
            self.riemannian_euclidean_dim
        )


# Convenience presets
def hyperbolic_only() -> GeometryConfig:
    """Hyperbolic embeddings only (for pure hierarchies)."""
    return GeometryConfig(
        use_riemannian=True,
        riemannian_hyperbolic_dim=768,
        riemannian_spherical_dim=0,
        riemannian_euclidean_dim=0,
        riemannian_hyperbolic_curvature=-1.0,
    )


def spherical_only() -> GeometryConfig:
    """Spherical embeddings only (for pure clusters)."""
    return GeometryConfig(
        use_riemannian=True,
        riemannian_hyperbolic_dim=0,
        riemannian_spherical_dim=768,
        riemannian_euclidean_dim=0,
        riemannian_spherical_curvature=1.0,
    )


def product_manifold() -> GeometryConfig:
    """Product manifold (all three spaces, balanced)."""
    return GeometryConfig(
        use_riemannian=True,
        riemannian_hyperbolic_dim=256,
        riemannian_spherical_dim=256,
        riemannian_euclidean_dim=256,
        riemannian_hyperbolic_curvature=-1.0,
        riemannian_spherical_curvature=1.0,
    )


def deep_hierarchy() -> GeometryConfig:
    """Optimized for deep hierarchies (e.g., taxonomies)."""
    return GeometryConfig(
        use_riemannian=True,
        riemannian_hyperbolic_dim=512,
        riemannian_spherical_dim=128,
        riemannian_euclidean_dim=128,
        riemannian_hyperbolic_curvature=-2.0,  # More room for deep trees
        riemannian_spherical_curvature=1.0,
    )


def tight_clusters() -> GeometryConfig:
    """Optimized for tight topic clusters."""
    return GeometryConfig(
        use_riemannian=True,
        riemannian_hyperbolic_dim=128,
        riemannian_spherical_dim=512,
        riemannian_euclidean_dim=128,
        riemannian_hyperbolic_curvature=-1.0,
        riemannian_spherical_curvature=2.0,  # Tighter clusters
    )


__all__ = [
    "GeometryConfig",
    "hyperbolic_only",
    "spherical_only",
    "product_manifold",
    "deep_hierarchy",
    "tight_clusters",
]
