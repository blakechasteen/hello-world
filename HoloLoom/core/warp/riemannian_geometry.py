"""
Riemannian Geometry for Semantic Space
======================================
Treat semantic embeddings as points on a curved manifold, not flat Euclidean space.

WHY THIS MATTERS:
- Semantic similarity is NOT Euclidean distance
- "Dog" → "Cat" → "Mammal" follows a geodesic (curved path)
- High-dimensional embeddings lie on low-dimensional manifold
- Curvature reveals semantic structure

BREAKTHROUGH INSIGHT:
Language embeddings naturally form curved manifolds. Treating them as
flat Euclidean space (dot products, L2 norms) loses critical geometric
structure. We need proper Riemannian tools:

1. Geodesic distance (not Euclidean)
2. Parallel transport (moving vectors along manifold)
3. Curvature analysis (detecting semantic clusters)
4. Exponential/logarithmic maps (manifold ↔ tangent space)

Mathematical Foundation:
-----------------------
Riemannian Manifold: (M, g)
- M: Smooth manifold (embedding space)
- g: Metric tensor (defines distances and angles)

For semantic embeddings, we use:
- Hyperbolic space (negative curvature) for hierarchies
- Spherical geometry (positive curvature) for clustering
- Mixed curvature spaces for complex semantics

Author: HoloLoom Differential Geometry Team
Date: 2025-11-03
"""

from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
import warnings


# ============================================================================
# Manifold Types
# ============================================================================

class ManifoldType(Enum):
    """Types of Riemannian manifolds."""

    EUCLIDEAN = "euclidean"        # Flat space (baseline)
    HYPERBOLIC = "hyperbolic"      # Negative curvature (hierarchies)
    SPHERICAL = "spherical"        # Positive curvature (clusters)
    PRODUCT = "product"            # Mixed curvature (H × S × E)


@dataclass
class ManifoldConfig:
    """Configuration for Riemannian manifold."""

    manifold_type: ManifoldType = ManifoldType.HYPERBOLIC
    curvature: float = -1.0         # Sectional curvature (K)
    dim: int = 384                  # Manifold dimension

    # For product manifolds
    hyperbolic_dim: int = 128       # Dimensions with K < 0
    spherical_dim: int = 128        # Dimensions with K > 0
    euclidean_dim: int = 128        # Dimensions with K = 0


# ============================================================================
# Hyperbolic Space (Poincaré Ball Model)
# ============================================================================

class HyperbolicSpace:
    """
    Poincaré ball model of hyperbolic space.

    The Poincaré ball is a conformal model where:
    - Points: Unit ball {x : ||x|| < 1}
    - Metric: ds² = 4/(1-||x||²)² dx²
    - Geodesics: Circular arcs orthogonal to boundary

    Perfect for hierarchical embeddings (trees, taxonomies).
    Distances grow exponentially, naturally encoding hierarchy.
    """

    def __init__(self, curvature: float = -1.0, eps: float = 1e-6):
        """
        Initialize hyperbolic space.

        Args:
            curvature: Sectional curvature K < 0 (typically -1)
            eps: Numerical stability epsilon
        """
        self.K = curvature
        self.eps = eps

        # Poincaré ball radius c = sqrt(-K)
        self.c = np.sqrt(-curvature) if curvature < 0 else 1.0

    def project(self, x: np.ndarray) -> np.ndarray:
        """
        Project point onto Poincaré ball.

        Ensures ||x|| < 1/c for numerical stability.

        Args:
            x: Point in R^n

        Returns:
            Projected point in Poincaré ball
        """
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        max_norm = (1.0 - self.eps) / self.c

        # Rescale if outside ball
        scale = np.where(norm > max_norm, max_norm / (norm + self.eps), 1.0)
        return x * scale

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute hyperbolic distance (geodesic).

        Formula:
            d_H(x, y) = (1/√c) × arcosh(1 + 2c² ||x-y||² / ((1-c²||x||²)(1-c²||y||²)))

        Args:
            x, y: Points in Poincaré ball

        Returns:
            Hyperbolic distance
        """
        # Project to ensure valid points
        x = self.project(x)
        y = self.project(y)

        # Squared norms
        x2 = np.sum(x * x, axis=-1)
        y2 = np.sum(y * y, axis=-1)
        xy = np.sum((x - y) * (x - y), axis=-1)

        # Avoid division by zero
        c2 = self.c * self.c
        denom = (1 - c2 * x2) * (1 - c2 * y2) + self.eps

        # Hyperbolic distance
        alpha = 1 + 2 * c2 * xy / denom
        alpha = np.maximum(alpha, 1.0 + self.eps)  # Numerical stability

        return np.arccosh(alpha) / self.c

    def exp_map(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Exponential map: move along geodesic from x in direction v.

        Maps tangent vector v at point x to a point on manifold.

        Args:
            x: Base point on manifold
            v: Tangent vector at x

        Returns:
            Point on manifold reached by geodesic
        """
        x = self.project(x)

        v_norm = np.linalg.norm(v, axis=-1, keepdims=True) + self.eps
        x_norm2 = np.sum(x * x, axis=-1, keepdims=True)

        # Lambda factor
        lambda_x = 2.0 / (1.0 - self.c * self.c * x_norm2)

        # Exponential map formula
        sqrt_c = self.c
        coef = np.tanh(sqrt_c * lambda_x * v_norm / 2) / (sqrt_c * v_norm)

        exp_map = self.mobius_add(x, coef * v)
        return self.project(exp_map)

    def log_map(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Logarithmic map: tangent vector from x to y.

        Inverse of exp_map. Maps manifold point y to tangent vector at x.

        Args:
            x: Base point
            y: Target point

        Returns:
            Tangent vector at x pointing toward y
        """
        x = self.project(x)
        y = self.project(y)

        diff = self.mobius_add(-x, y)
        diff_norm = np.linalg.norm(diff, axis=-1, keepdims=True) + self.eps

        x_norm2 = np.sum(x * x, axis=-1, keepdims=True)
        lambda_x = 2.0 / (1.0 - self.c * self.c * x_norm2)

        sqrt_c = self.c
        atanh_arg = np.clip(sqrt_c * diff_norm, -1 + self.eps, 1 - self.eps)

        return (2.0 / (sqrt_c * lambda_x)) * np.arctanh(atanh_arg) * (diff / diff_norm)

    def mobius_add(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Möbius addition (hyperbolic addition in Poincaré ball).

        Args:
            x, y: Points in Poincaré ball

        Returns:
            x ⊕ y in hyperbolic space
        """
        c2 = self.c * self.c

        x2 = np.sum(x * x, axis=-1, keepdims=True)
        y2 = np.sum(y * y, axis=-1, keepdims=True)
        xy = np.sum(x * y, axis=-1, keepdims=True)

        num = (1 + 2 * c2 * xy + c2 * y2) * x + (1 - c2 * x2) * y
        denom = 1 + 2 * c2 * xy + c2 * c2 * x2 * y2 + self.eps

        return num / denom

    def parallel_transport(
        self,
        x: np.ndarray,
        y: np.ndarray,
        v: np.ndarray
    ) -> np.ndarray:
        """
        Parallel transport vector v from x to y.

        Moves tangent vector along geodesic while preserving inner product.
        Critical for comparing gradients at different points.

        Args:
            x: Source point
            y: Target point
            v: Tangent vector at x

        Returns:
            Parallel-transported vector at y
        """
        x = self.project(x)
        y = self.project(y)

        # Log map: tangent direction from x to y
        log_xy = self.log_map(x, y)
        log_norm = np.linalg.norm(log_xy, axis=-1, keepdims=True) + self.eps

        # Gyration (parallel transport in Poincaré ball)
        lambda_x = 2.0 / (1.0 - self.c * self.c * np.sum(x * x, axis=-1, keepdims=True))
        lambda_y = 2.0 / (1.0 - self.c * self.c * np.sum(y * y, axis=-1, keepdims=True))

        scale = lambda_x / lambda_y
        return scale * v


# ============================================================================
# Spherical Geometry (Unit Sphere)
# ============================================================================

class SphericalSpace:
    """
    Unit sphere S^n embedded in R^{n+1}.

    Spherical geometry for clustered semantic spaces.
    Natural for normalized embeddings (cosine similarity).

    - Points: Unit sphere {x : ||x|| = 1}
    - Geodesics: Great circles
    - Distance: Angular distance
    """

    def __init__(self, curvature: float = 1.0, eps: float = 1e-6):
        """
        Initialize spherical space.

        Args:
            curvature: Sectional curvature K > 0 (typically 1)
            eps: Numerical stability epsilon
        """
        self.K = curvature
        self.eps = eps
        self.radius = 1.0 / np.sqrt(curvature) if curvature > 0 else 1.0

    def project(self, x: np.ndarray) -> np.ndarray:
        """
        Project point onto unit sphere.

        Args:
            x: Point in R^n

        Returns:
            Normalized point on sphere
        """
        norm = np.linalg.norm(x, axis=-1, keepdims=True) + self.eps
        return self.radius * x / norm

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute spherical distance (great circle arc length).

        Formula: d_S(x, y) = arccos(<x, y>) / √K

        Args:
            x, y: Points on sphere

        Returns:
            Angular distance
        """
        x = self.project(x)
        y = self.project(y)

        # Cosine similarity (clipped for numerical stability)
        cos_angle = np.sum(x * y, axis=-1)
        cos_angle = np.clip(cos_angle, -1.0 + self.eps, 1.0 - self.eps)

        return np.arccos(cos_angle) / np.sqrt(self.K)

    def exp_map(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Exponential map on sphere.

        Args:
            x: Base point on sphere
            v: Tangent vector at x

        Returns:
            Point reached by geodesic
        """
        x = self.project(x)

        v_norm = np.linalg.norm(v, axis=-1, keepdims=True) + self.eps
        sqrt_K = np.sqrt(self.K)

        # Exponential map formula
        exp_map = np.cos(sqrt_K * v_norm) * x + np.sin(sqrt_K * v_norm) * (v / v_norm)
        return self.project(exp_map)

    def log_map(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Logarithmic map on sphere.

        Args:
            x: Base point
            y: Target point

        Returns:
            Tangent vector from x to y
        """
        x = self.project(x)
        y = self.project(y)

        xy = np.sum(x * y, axis=-1, keepdims=True)
        xy = np.clip(xy, -1.0 + self.eps, 1.0 - self.eps)

        dist = np.arccos(xy) / np.sqrt(self.K)

        # Tangent direction
        diff = y - xy * x
        diff_norm = np.linalg.norm(diff, axis=-1, keepdims=True) + self.eps

        return dist * (diff / diff_norm)

    def parallel_transport(
        self,
        x: np.ndarray,
        y: np.ndarray,
        v: np.ndarray
    ) -> np.ndarray:
        """
        Parallel transport on sphere.

        Args:
            x: Source point
            y: Target point
            v: Tangent vector at x

        Returns:
            Parallel-transported vector at y
        """
        x = self.project(x)
        y = self.project(y)

        # Schild's ladder approximation for parallel transport
        xy = np.sum(x * y, axis=-1, keepdims=True)

        # Project v onto tangent space at y
        v_transported = v - xy * (v - np.sum(v * x, axis=-1, keepdims=True) * y)
        return v_transported


# ============================================================================
# Product Manifold (Mixed Curvature)
# ============================================================================

class ProductManifold:
    """
    Product manifold: H^n₁ × S^n₂ × E^n₃

    Combines multiple curvatures for rich semantic structure:
    - Hyperbolic: Hierarchical concepts (K < 0)
    - Spherical: Clustered concepts (K > 0)
    - Euclidean: Linear features (K = 0)

    This is closest to reality: semantic space has mixed structure.
    """

    def __init__(self, config: ManifoldConfig):
        """
        Initialize product manifold.

        Args:
            config: Manifold configuration
        """
        self.config = config

        # Component spaces
        self.hyperbolic = HyperbolicSpace(curvature=config.curvature)
        self.spherical = SphericalSpace(curvature=abs(config.curvature))

        # Dimension splits
        self.h_dim = config.hyperbolic_dim
        self.s_dim = config.spherical_dim
        self.e_dim = config.euclidean_dim

    def split(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split vector into component subspaces."""
        h_end = self.h_dim
        s_end = h_end + self.s_dim

        return x[..., :h_end], x[..., h_end:s_end], x[..., s_end:]

    def combine(
        self,
        h: np.ndarray,
        s: np.ndarray,
        e: np.ndarray
    ) -> np.ndarray:
        """Combine subspace vectors."""
        return np.concatenate([h, s, e], axis=-1)

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute product manifold distance.

        d_product² = d_H² + d_S² + d_E²

        Args:
            x, y: Points on product manifold

        Returns:
            Distance
        """
        x_h, x_s, x_e = self.split(x)
        y_h, y_s, y_e = self.split(y)

        # Component distances
        d_h = self.hyperbolic.distance(x_h, y_h)
        d_s = self.spherical.distance(x_s, y_s)
        d_e = np.linalg.norm(x_e - y_e, axis=-1)

        # Pythagorean sum
        return np.sqrt(d_h**2 + d_s**2 + d_e**2)

    def exp_map(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Exponential map on product manifold."""
        x_h, x_s, x_e = self.split(x)
        v_h, v_s, v_e = self.split(v)

        exp_h = self.hyperbolic.exp_map(x_h, v_h)
        exp_s = self.spherical.exp_map(x_s, v_s)
        exp_e = x_e + v_e  # Euclidean

        return self.combine(exp_h, exp_s, exp_e)

    def log_map(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Logarithmic map on product manifold."""
        x_h, x_s, x_e = self.split(x)
        y_h, y_s, y_e = self.split(y)

        log_h = self.hyperbolic.log_map(x_h, y_h)
        log_s = self.spherical.log_map(x_s, y_s)
        log_e = y_e - x_e  # Euclidean

        return self.combine(log_h, log_s, log_e)

    def parallel_transport(
        self,
        x: np.ndarray,
        y: np.ndarray,
        v: np.ndarray
    ) -> np.ndarray:
        """Parallel transport on product manifold."""
        x_h, x_s, x_e = self.split(x)
        y_h, y_s, y_e = self.split(y)
        v_h, v_s, v_e = self.split(v)

        pt_h = self.hyperbolic.parallel_transport(x_h, y_h, v_h)
        pt_s = self.spherical.parallel_transport(x_s, y_s, v_s)
        pt_e = v_e  # Euclidean (identity)

        return self.combine(pt_h, pt_s, pt_e)


# ============================================================================
# Manifold Factory
# ============================================================================

def create_manifold(config: ManifoldConfig):
    """
    Factory function to create manifold.

    Args:
        config: Manifold configuration

    Returns:
        Manifold instance (Hyperbolic, Spherical, or Product)
    """
    if config.manifold_type == ManifoldType.HYPERBOLIC:
        return HyperbolicSpace(curvature=config.curvature)

    elif config.manifold_type == ManifoldType.SPHERICAL:
        return SphericalSpace(curvature=config.curvature)

    elif config.manifold_type == ManifoldType.PRODUCT:
        return ProductManifold(config)

    elif config.manifold_type == ManifoldType.EUCLIDEAN:
        # Fallback: standard Euclidean space
        warnings.warn("Using Euclidean space (no Riemannian structure)")
        return None

    else:
        raise ValueError(f"Unknown manifold type: {config.manifold_type}")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'ManifoldType',
    'ManifoldConfig',
    'HyperbolicSpace',
    'SphericalSpace',
    'ProductManifold',
    'create_manifold',
]
