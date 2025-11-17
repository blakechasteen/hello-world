"""
HoloLoom Cross-Modal Fusion
============================
Strategies for combining embeddings across modalities.

This module implements fusion strategies to combine embeddings
from different modalities (text, image, audio, video) into
unified representations.

Fusion Strategies:
1. Early Fusion: Concatenate raw features before encoding
2. Late Fusion: Weighted combination of modal embeddings
3. Hybrid Fusion: Learnable cross-attention between modalities

Philosophy:
All modalities share a common 768D embedding space. Fusion
creates representations that capture cross-modal semantics,
enabling queries like "find videos about cats" to match both
visual content and audio transcriptions.
"""

import logging
import warnings
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import numpy as np

# Import from base module
from HoloLoom.multimodal.base import (
    Modality,
    ModalEmbedding,
    MultiModalEmbedding,
    FusionConfig,
    normalize_embedding
)

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAVE_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    _HAVE_TORCH = False
    warnings.warn("torch not available. Learnable fusion unavailable.")


# ============================================================================
# Fusion Strategies
# ============================================================================

class LateFusion:
    """
    Late fusion: Weighted combination of modal embeddings.

    Simple and effective strategy that combines embeddings
    using configurable weights. No learnable parameters.

    Example:
        fuser = LateFusion(weights={Modality.TEXT: 0.5, Modality.IMAGE: 0.5})
        fused = fuser.fuse(embeddings)
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """
        Initialize late fusion.

        Args:
            config: Fusion configuration with modal weights
        """
        self.config = config or FusionConfig(method="late")
        self.weights = self.config.weights

    def fuse(
        self,
        embeddings: Dict[Modality, List[float]],
        **kwargs
    ) -> List[float]:
        """
        Fuse modal embeddings with weighted averaging.

        Args:
            embeddings: Dict mapping modality to 768D embedding
            **kwargs: Optional per-call weights override

        Returns:
            Fused 768D embedding
        """
        if not embeddings:
            return [0.0] * 768

        # Use override weights if provided
        weights = kwargs.get("weights", self.weights)

        # Normalize weights to sum to 1.0
        total_weight = sum(
            weights.get(mod, 0.0)
            for mod in embeddings.keys()
        )

        if total_weight == 0:
            # Uniform weighting if no weights specified
            total_weight = len(embeddings)
            weights = {mod: 1.0 for mod in embeddings.keys()}

        # Weighted sum
        fused = np.zeros(768)
        for modality, embedding in embeddings.items():
            weight = weights.get(modality, 0.0) / total_weight
            fused += weight * np.array(embedding)

        # Normalize if configured
        if self.config.normalize:
            fused = fused / (np.linalg.norm(fused) + 1e-9)

        return fused.tolist()


class EarlyFusion:
    """
    Early fusion: Concatenate features before final projection.

    Concatenates modal embeddings and projects back to 768D.
    Uses a simple linear projection (can be replaced with MLP).

    Note: This is a simplified implementation. True early fusion
    would happen before encoding (at feature level), but since
    our encoders already produce 768D, we simulate by concatenating
    and projecting.

    Example:
        fuser = EarlyFusion()
        fused = fuser.fuse(embeddings)
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """
        Initialize early fusion.

        Args:
            config: Fusion configuration
        """
        self.config = config or FusionConfig(method="early")

    def fuse(
        self,
        embeddings: Dict[Modality, List[float]],
        **kwargs
    ) -> List[float]:
        """
        Fuse modal embeddings with concatenation + projection.

        Args:
            embeddings: Dict mapping modality to 768D embedding
            **kwargs: Additional options

        Returns:
            Fused 768D embedding
        """
        if not embeddings:
            return [0.0] * 768

        # Concatenate all embeddings
        concat = []
        for embedding in embeddings.values():
            concat.extend(embedding)

        # Convert to numpy
        concat_arr = np.array(concat)

        # Project back to 768D
        # Simple approach: reshape and average
        n_modalities = len(embeddings)
        concat_arr = concat_arr.reshape(n_modalities, 768)
        fused = np.mean(concat_arr, axis=0)

        # Normalize if configured
        if self.config.normalize:
            fused = fused / (np.linalg.norm(fused) + 1e-9)

        return fused.tolist()


class HybridFusion:
    """
    Hybrid fusion: Cross-attention between modalities.

    Uses multi-head attention to compute cross-modal interactions
    before combining. This is a learnable fusion strategy.

    Architecture:
    1. Each modal embedding → query, key, value
    2. Cross-attention between all modalities
    3. Concatenate attended outputs
    4. Project to 768D

    Note: Requires torch. Falls back to late fusion if unavailable.

    Example:
        fuser = HybridFusion()
        fused = fuser.fuse(embeddings)
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """
        Initialize hybrid fusion.

        Args:
            config: Fusion configuration
        """
        self.config = config or FusionConfig(method="hybrid")
        self._attention = None

        # Initialize attention module if torch available
        if _HAVE_TORCH:
            self._attention = CrossModalAttention(
                embed_dim=768,
                num_heads=8
            )
        else:
            logger.warning(
                "torch unavailable - hybrid fusion will use late fusion"
            )

    def fuse(
        self,
        embeddings: Dict[Modality, List[float]],
        **kwargs
    ) -> List[float]:
        """
        Fuse modal embeddings with cross-attention.

        Args:
            embeddings: Dict mapping modality to 768D embedding
            **kwargs: Additional options

        Returns:
            Fused 768D embedding
        """
        if not embeddings:
            return [0.0] * 768

        # Fallback to late fusion if torch unavailable
        if self._attention is None:
            fuser = LateFusion(self.config)
            return fuser.fuse(embeddings, **kwargs)

        # Convert to torch tensors
        modal_tensors = {}
        for modality, embedding in embeddings.items():
            tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
            modal_tensors[modality] = tensor

        # Apply cross-attention
        with torch.no_grad():
            fused_tensor = self._attention(modal_tensors)

        # Convert back to list
        fused = fused_tensor.squeeze(0).numpy().tolist()

        # Normalize if configured
        if self.config.normalize:
            fused_arr = np.array(fused)
            fused_arr = fused_arr / (np.linalg.norm(fused_arr) + 1e-9)
            fused = fused_arr.tolist()

        return fused


# ============================================================================
# Cross-Modal Attention (for Hybrid Fusion)
# ============================================================================

if _HAVE_TORCH:
    class CrossModalAttention(nn.Module):
        """
        Cross-modal attention module.

        Computes attention between different modalities to create
        a fused representation that captures cross-modal semantics.
        """

        def __init__(self, embed_dim: int = 768, num_heads: int = 8):
            """
            Initialize cross-modal attention.

            Args:
                embed_dim: Embedding dimension (768)
                num_heads: Number of attention heads
            """
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads

            # Multi-head attention
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True
            )

            # Output projection
            self.out_proj = nn.Linear(embed_dim, embed_dim)

        def forward(
            self,
            modal_tensors: Dict[Modality, torch.Tensor]
        ) -> torch.Tensor:
            """
            Forward pass.

            Args:
                modal_tensors: Dict mapping modality to tensor (batch, embed_dim)

            Returns:
                Fused tensor (batch, embed_dim)
            """
            # Stack modalities (batch, n_modalities, embed_dim)
            modalities = list(modal_tensors.values())
            stacked = torch.cat(modalities, dim=0).unsqueeze(0)

            # Self-attention across modalities
            attended, _ = self.attn(stacked, stacked, stacked)

            # Mean pooling across modalities
            pooled = torch.mean(attended, dim=1)

            # Output projection
            output = self.out_proj(pooled)

            return output


# ============================================================================
# Multi-Modal Fuser (Unified Interface)
# ============================================================================

class MultiModalFuser:
    """
    Unified interface for multi-modal fusion.

    Supports multiple fusion strategies and can switch between them.

    Example:
        fuser = MultiModalFuser(method="late")
        fused_emb = fuser.fuse_embeddings(modal_embeddings)
    """

    def __init__(
        self,
        method: str = "late",
        config: Optional[FusionConfig] = None
    ):
        """
        Initialize multi-modal fuser.

        Args:
            method: Fusion method ("early", "late", "hybrid")
            config: Optional fusion configuration
        """
        self.method = method
        self.config = config or FusionConfig(method=method)

        # Initialize strategy
        if method == "late":
            self.strategy = LateFusion(self.config)
        elif method == "early":
            self.strategy = EarlyFusion(self.config)
        elif method == "hybrid":
            self.strategy = HybridFusion(self.config)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

        logger.info(f"Initialized {method} fusion strategy")

    def fuse_embeddings(
        self,
        embeddings: Union[
            Dict[Modality, List[float]],
            List[ModalEmbedding]
        ],
        **kwargs
    ) -> MultiModalEmbedding:
        """
        Fuse multiple modal embeddings.

        Args:
            embeddings: Either dict of modality→embedding or list of ModalEmbeddings
            **kwargs: Additional fusion options

        Returns:
            MultiModalEmbedding with fused representation
        """
        # Normalize input format
        if isinstance(embeddings, list):
            # Convert list of ModalEmbeddings to dict
            emb_dict = {
                emb.modality: emb.embedding
                for emb in embeddings
            }
        else:
            emb_dict = embeddings

        # Fuse embeddings
        fused = self.strategy.fuse(emb_dict, **kwargs)

        # Create MultiModalEmbedding
        result = MultiModalEmbedding(
            embeddings=emb_dict,
            fused_embedding=fused,
            metadata={
                "fusion_method": self.method,
                "modalities": [mod.value for mod in emb_dict.keys()],
                "num_modalities": len(emb_dict)
            }
        )

        return result

    def compute_similarity(
        self,
        embedding1: MultiModalEmbedding,
        embedding2: MultiModalEmbedding
    ) -> float:
        """
        Compute similarity between two multi-modal embeddings.

        Args:
            embedding1: First multi-modal embedding
            embedding2: Second multi-modal embedding

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Use fused embeddings if available
        emb1 = embedding1.fused_embedding
        emb2 = embedding2.fused_embedding

        if emb1 is None or emb2 is None:
            return 0.0

        # Cosine similarity
        v1 = np.array(emb1)
        v2 = np.array(emb2)

        similarity = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
        )

        return float(max(0.0, min(1.0, similarity)))


# ============================================================================
# Factory Function
# ============================================================================

def create_fuser(
    method: str = "late",
    weights: Optional[Dict[Modality, float]] = None,
    **kwargs
) -> MultiModalFuser:
    """
    Factory function to create multi-modal fuser.

    Args:
        method: Fusion method ("early", "late", "hybrid")
        weights: Optional modal weights for late fusion
        **kwargs: Additional config options

    Returns:
        MultiModalFuser instance
    """
    config = FusionConfig(
        method=method,
        weights=weights or FusionConfig().weights,
        **kwargs
    )
    return MultiModalFuser(method=method, config=config)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Multi-Modal Fusion Demo")
    print("=" * 80 + "\n")

    # Create sample embeddings
    embeddings = {
        Modality.TEXT: np.random.randn(768).tolist(),
        Modality.IMAGE: np.random.randn(768).tolist(),
        Modality.AUDIO: np.random.randn(768).tolist()
    }

    print("Testing fusion strategies:\n")

    # Late fusion
    late_fuser = create_fuser(method="late")
    late_result = late_fuser.fuse_embeddings(embeddings)
    print(f"Late fusion:")
    print(f"  Modalities: {late_result.metadata['modalities']}")
    print(f"  Fused dim: {len(late_result.fused_embedding)}")

    # Early fusion
    early_fuser = create_fuser(method="early")
    early_result = early_fuser.fuse_embeddings(embeddings)
    print(f"\nEarly fusion:")
    print(f"  Modalities: {early_result.metadata['modalities']}")
    print(f"  Fused dim: {len(early_result.fused_embedding)}")

    # Hybrid fusion
    hybrid_fuser = create_fuser(method="hybrid")
    hybrid_result = hybrid_fuser.fuse_embeddings(embeddings)
    print(f"\nHybrid fusion:")
    print(f"  Modalities: {hybrid_result.metadata['modalities']}")
    print(f"  Fused dim: {len(hybrid_result.fused_embedding)}")

    # Similarity
    similarity = late_fuser.compute_similarity(late_result, early_result)
    print(f"\nSimilarity (late vs early): {similarity:.3f}")

    print("\n✓ Fusion module ready!")
