"""
Riemannian Matryoshka Embeddings
=================================
Matryoshka embeddings with Riemannian manifold structure for true semantic distances.

This module wraps MatryoshkaEmbeddings and adds geodesic distance computation
using product manifold H×S×E (hyperbolic × spherical × euclidean).

WHY THIS MATTERS:
- Semantic similarity is NOT Euclidean distance
- Hierarchies live in hyperbolic space (tree structure)
- Clusters live on spheres (normalized cosine similarity)
- Linear features remain in Euclidean space

Author: HoloLoom Mathematical Moonshot Team
Date: 2025-11-03
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, List
import numpy as np

# Import base embedder
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

# Import Riemannian geometry for geodesic distance support
try:
    from HoloLoom.warp.riemannian_geometry import (
        ProductManifold, ManifoldConfig, ManifoldType
    )
    _HAVE_RIEMANNIAN = True
except ImportError:
    ProductManifold = None
    ManifoldConfig = None
    ManifoldType = None
    _HAVE_RIEMANNIAN = False
    warnings.warn(
        "Riemannian geometry module not available. "
        "RiemannianMatryoshka will fall back to Euclidean distances."
    )


@dataclass
class RiemannianMatryoshka:
    """
    Matryoshka embeddings with Riemannian manifold structure.

    Wraps MatryoshkaEmbeddings and adds geodesic distance computation
    using product manifold H×S×E (hyperbolic × spherical × euclidean).

    WHY THIS MATTERS:
    - Semantic similarity is NOT Euclidean distance
    - Hierarchies live in hyperbolic space (tree structure)
    - Clusters live on spheres (normalized cosine similarity)
    - Linear features remain in Euclidean space

    When to use:
    - use_riemannian=True: Compute true semantic distance via geodesics
    - use_riemannian=False: Standard Euclidean L2 distance (default)

    Backward compatible: Falls back gracefully if riemannian_geometry unavailable.

    Example:
    ```python
    from HoloLoom.embedding.riemannian_matryoshka import RiemannianMatryoshka

    # Create Riemannian embedder
    embedder = RiemannianMatryoshka(
        use_riemannian=True,
        hyperbolic_dim=128,  # Hierarchical concepts
        spherical_dim=128,   # Clustered concepts
        euclidean_dim=128    # Linear features
    )

    # Encode texts
    texts = ["mammal", "dog", "beagle"]
    embs = embedder.encode(texts)

    # Compute geodesic distance (not Euclidean!)
    dist = embedder.distance(embs[0], embs[1])
    ```
    """

    base_embedder: MatryoshkaEmbeddings = field(default_factory=lambda: MatryoshkaEmbeddings())
    use_riemannian: bool = False

    # Manifold configuration
    hyperbolic_dim: int = 128  # Hierarchical concepts (K < 0)
    spherical_dim: int = 128   # Clustered concepts (K > 0)
    euclidean_dim: int = 128   # Linear features (K = 0)

    # Curvatures
    hyperbolic_curvature: float = -1.0  # Negative curvature for hierarchies
    spherical_curvature: float = 1.0    # Positive curvature for clusters

    def __post_init__(self):
        """Initialize Riemannian manifold if enabled."""
        self.manifold = None

        if self.use_riemannian:
            if not _HAVE_RIEMANNIAN:
                warnings.warn(
                    "Riemannian geometry not available. Falling back to Euclidean distances. "
                    "To enable: ensure HoloLoom.warp.riemannian_geometry is accessible."
                )
                self.use_riemannian = False
                return

            # Create product manifold configuration
            total_dim = self.hyperbolic_dim + self.spherical_dim + self.euclidean_dim

            config = ManifoldConfig(
                manifold_type=ManifoldType.PRODUCT,
                curvature=self.hyperbolic_curvature,
                dim=total_dim,
                hyperbolic_dim=self.hyperbolic_dim,
                spherical_dim=self.spherical_dim,
                euclidean_dim=self.euclidean_dim
            )

            # Create product manifold
            self.manifold = ProductManifold(config)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using base embedder.

        Args:
            texts: List of text strings

        Returns:
            Matrix of embeddings
        """
        return self.base_embedder.encode(texts)

    def encode_scales(
        self,
        texts: List[str],
        size: Optional[int] = None
    ) -> Union[Dict[int, np.ndarray], np.ndarray]:
        """
        Encode texts at one or all scales (delegates to base embedder).

        Args:
            texts: List of text strings
            size: Optional specific size to return

        Returns:
            If size specified: Matrix of embeddings
            If size is None: Dict mapping size → embedding matrix
        """
        return self.base_embedder.encode_scales(texts, size=size)

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute distance between embeddings.

        Uses geodesic distance if use_riemannian=True, else Euclidean L2.

        Args:
            x: First embedding vector
            y: Second embedding vector

        Returns:
            Distance (geodesic or Euclidean)
        """
        if self.use_riemannian and self.manifold is not None:
            # Ensure vectors are 1D
            x_1d = x.flatten() if x.ndim > 1 else x
            y_1d = y.flatten() if y.ndim > 1 else y

            # Compute geodesic distance on product manifold
            return self.manifold.distance(x_1d, y_1d)
        else:
            # Standard Euclidean L2 distance
            return float(np.linalg.norm(x - y))

    def pairwise_distances(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute pairwise distances between sets of embeddings.

        Args:
            X: Matrix of embeddings (n_samples × dim)
            Y: Optional second matrix. If None, computes X vs X

        Returns:
            Distance matrix (n_samples_X × n_samples_Y)
        """
        if Y is None:
            Y = X

        n_x = X.shape[0]
        n_y = Y.shape[0]

        distances = np.zeros((n_x, n_y))

        for i in range(n_x):
            for j in range(n_y):
                distances[i, j] = self.distance(X[i], Y[j])

        return distances

    def exp_map(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Exponential map: move along geodesic from x in direction v.

        Only available if use_riemannian=True.

        Args:
            x: Base point on manifold
            v: Tangent vector at x

        Returns:
            Point on manifold reached by geodesic
        """
        if not self.use_riemannian or self.manifold is None:
            raise ValueError(
                "Exponential map only available with use_riemannian=True "
                "and Riemannian geometry module."
            )

        return self.manifold.exp_map(x, v)

    def log_map(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Logarithmic map: tangent vector from x to y.

        Only available if use_riemannian=True.

        Args:
            x: Base point
            y: Target point

        Returns:
            Tangent vector at x pointing toward y
        """
        if not self.use_riemannian or self.manifold is None:
            raise ValueError(
                "Logarithmic map only available with use_riemannian=True "
                "and Riemannian geometry module."
            )

        return self.manifold.log_map(x, y)

    def parallel_transport(
        self,
        x: np.ndarray,
        y: np.ndarray,
        v: np.ndarray
    ) -> np.ndarray:
        """
        Parallel transport vector v from x to y.

        Only available if use_riemannian=True.

        Args:
            x: Source point
            y: Target point
            v: Tangent vector at x

        Returns:
            Parallel-transported vector at y
        """
        if not self.use_riemannian or self.manifold is None:
            raise ValueError(
                "Parallel transport only available with use_riemannian=True "
                "and Riemannian geometry module."
            )

        return self.manifold.parallel_transport(x, y, v)


# ============================================================================
# Factory Function
# ============================================================================

def create_riemannian_embedder(
    use_riemannian: bool = False,
    sizes: List[int] = [768],
    hyperbolic_dim: int = 256,
    spherical_dim: int = 256,
    euclidean_dim: int = 256,
    base_model_name: Optional[str] = None
) -> RiemannianMatryoshka:
    """
    Factory function to create Riemannian embedder.

    Args:
        use_riemannian: Enable Riemannian manifold structure
        sizes: Matryoshka embedding sizes
        hyperbolic_dim: Dimension for hierarchical concepts
        spherical_dim: Dimension for clustered concepts
        euclidean_dim: Dimension for linear features
        base_model_name: Optional base model for embeddings

    Returns:
        RiemannianMatryoshka embedder
    """
    base_embedder = MatryoshkaEmbeddings(
        sizes=sizes,
        base_model_name=base_model_name
    )

    return RiemannianMatryoshka(
        base_embedder=base_embedder,
        use_riemannian=use_riemannian,
        hyperbolic_dim=hyperbolic_dim,
        spherical_dim=spherical_dim,
        euclidean_dim=euclidean_dim
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'RiemannianMatryoshka',
    'create_riemannian_embedder',
]
