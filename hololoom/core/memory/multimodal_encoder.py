#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodal Encoder
==================

Multi-representation image encoding for HoloLoom's visual memory.

Encodes images into multiple complementary representations:
1. CLIP embeddings (512D) - Semantic image-text matching
2. Structural features - Low-level visual properties
3. Caption embeddings (optional) - Text representation of image content

This enables flexible retrieval:
- Text → Image (CLIP text-image similarity)
- Image → Image (CLIP image-image similarity)
- Structural queries (color, layout, aspect ratio)

Usage:
    encoder = MultimodalEncoder(use_clip=True)
    embeddings = await encoder.encode_image(image_array)
    # Returns: {'clip': array(512), 'structural': {...}}
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Optional dependencies
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    import clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    clip = None
    torch = None


@dataclass
class ImageEmbeddings:
    """
    Multi-representation image embeddings.

    Attributes:
        clip: CLIP visual embedding (512D, optional)
        structural: Structural features dict
        caption: Text embedding of caption (optional)
        metadata: Additional metadata
    """
    clip: Optional[np.ndarray] = None
    structural: Dict[str, float] = None
    caption: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.structural is None:
            self.structural = {}
        if self.metadata is None:
            self.metadata = {}


class MultimodalEncoder:
    """
    Multi-representation image encoder.

    Combines multiple encoding strategies for comprehensive image understanding:
    - CLIP: Semantic embedding for image-text matching
    - Structural: Color, brightness, layout, aspect ratio
    - Caption: Text embedding (optional, for captioned images)

    Performance:
        CLIP encoding: ~50-100ms (CPU)
        Structural features: <5ms
        Total: ~100-150ms per image

    Usage:
        encoder = MultimodalEncoder(use_clip=True)
        embeddings = await encoder.encode_image(image_rgb)
    """

    def __init__(
        self,
        use_clip: bool = True,
        clip_model_name: str = "ViT-B/32",
        device: str = "cpu"
    ):
        """
        Initialize multimodal encoder.

        Args:
            use_clip: Enable CLIP embeddings (requires clip package)
            clip_model_name: CLIP model variant ("ViT-B/32", "ViT-B/16", etc.)
            device: Compute device ("cpu", "cuda")
        """
        self.use_clip = use_clip and CLIP_AVAILABLE
        self.clip_model_name = clip_model_name
        self.device = device

        # CLIP model (lazy-loaded)
        self.clip_model = None
        self.clip_preprocess = None

        # Stats
        self.stats = {
            'total_encoded': 0,
            'clip_enabled': self.use_clip
        }

    async def initialize(self):
        """Initialize CLIP model (lazy initialization)."""
        if self.use_clip and self.clip_model is None:
            if not CLIP_AVAILABLE:
                print("⚠ CLIP not available. Install: pip install openai-clip")
                self.use_clip = False
                return

            # Load CLIP model
            self.clip_model, self.clip_preprocess = clip.load(
                self.clip_model_name,
                device=self.device
            )
            self.clip_model.eval()

    async def encode_image(
        self,
        image: np.ndarray,
        caption: Optional[str] = None
    ) -> ImageEmbeddings:
        """
        Encode image to multiple representations.

        Args:
            image: RGB image array (H, W, 3) with values 0-255
            caption: Optional text caption for caption embedding

        Returns:
            ImageEmbeddings with CLIP, structural, and optional caption embeddings

        Raises:
            ValueError: If image is not valid RGB array
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape {image.shape}")

        # Ensure initialized
        await self.initialize()

        embeddings = ImageEmbeddings()

        # CLIP embedding (512D)
        if self.use_clip and self.clip_model is not None:
            embeddings.clip = await self._encode_clip(image)

        # Structural features
        embeddings.structural = self._extract_structural_features(image)

        # Caption embedding (optional)
        if caption and self.use_clip and self.clip_model is not None:
            embeddings.caption = await self._encode_text_clip(caption)

        # Metadata
        h, w = image.shape[:2]
        embeddings.metadata = {
            'dimensions': (w, h),
            'encoding_method': 'clip' if self.use_clip else 'structural_only'
        }

        self.stats['total_encoded'] += 1

        return embeddings

    async def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text with CLIP.

        Args:
            text: Text query

        Returns:
            512D CLIP text embedding

        Raises:
            RuntimeError: If CLIP not available
        """
        if not self.use_clip or self.clip_model is None:
            raise RuntimeError("CLIP not available. Cannot encode text.")

        await self.initialize()
        return await self._encode_text_clip(text)

    async def _encode_clip(self, image: np.ndarray) -> np.ndarray:
        """
        Encode image with CLIP (512D).

        Args:
            image: RGB numpy array (H, W, 3)

        Returns:
            512D normalized CLIP embedding
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL required for CLIP encoding. Install: pip install Pillow")

        # Convert to PIL Image
        pil_image = Image.fromarray(image.astype('uint8'))

        # Preprocess for CLIP
        image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)

        # Encode with CLIP
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            # L2 normalization for cosine similarity
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()[0]

    async def _encode_text_clip(self, text: str) -> np.ndarray:
        """
        Encode text with CLIP (512D).

        Args:
            text: Text string

        Returns:
            512D normalized CLIP text embedding
        """
        # Tokenize text
        text_input = clip.tokenize([text]).to(self.device)

        # Encode with CLIP
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_input)
            # L2 normalization
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy()[0]

    def _extract_structural_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract low-level structural features.

        Features extracted:
        1. Color statistics:
           - mean_r, mean_g, mean_b: Mean RGB channel values (0-1)
           - color_variance: Variance of RGB values
           - dominant_channel: Which channel (R/G/B) has highest mean

        2. Brightness/Contrast:
           - brightness: Mean pixel intensity (0-1)
           - contrast: Standard deviation of brightness

        3. Layout:
           - aspect_ratio: Width / height ratio
           - is_portrait: True if height > width
           - is_landscape: True if width > height
           - is_square: True if aspect ratio close to 1.0

        Args:
            image: RGB numpy array (H, W, 3), values 0-255

        Returns:
            Dictionary of feature name → float value
        """
        features = {}

        # Normalize to 0-1
        image_norm = image.astype(np.float32) / 255.0

        # Color statistics
        features['mean_r'] = float(image_norm[:, :, 0].mean())
        features['mean_g'] = float(image_norm[:, :, 1].mean())
        features['mean_b'] = float(image_norm[:, :, 2].mean())

        # Color variance (how colorful?)
        color_variance = image_norm.std()
        features['color_variance'] = float(color_variance)

        # Dominant channel
        channel_means = [features['mean_r'], features['mean_g'], features['mean_b']]
        dominant_idx = np.argmax(channel_means)
        features['dominant_channel'] = float(dominant_idx)  # 0=R, 1=G, 2=B

        # Brightness and contrast
        brightness = image_norm.mean()
        features['brightness'] = float(brightness)

        contrast = image_norm.std()
        features['contrast'] = float(contrast)

        # Layout features
        h, w = image.shape[:2]
        aspect_ratio = w / h
        features['aspect_ratio'] = float(aspect_ratio)

        features['is_portrait'] = float(h > w)
        features['is_landscape'] = float(w > h)
        features['is_square'] = float(abs(aspect_ratio - 1.0) < 0.1)

        # Edge density (simple Sobel approximation)
        # Horizontal edges
        h_edges = np.abs(image_norm[1:, :, :] - image_norm[:-1, :, :]).mean()
        features['edge_density_h'] = float(h_edges)

        # Vertical edges
        v_edges = np.abs(image_norm[:, 1:, :] - image_norm[:, :-1, :]).mean()
        features['edge_density_v'] = float(v_edges)

        # Total edge density
        features['edge_density'] = float((h_edges + v_edges) / 2.0)

        return features

    def get_stats(self) -> Dict:
        """Get encoding statistics."""
        return {
            'total_encoded': self.stats['total_encoded'],
            'clip_enabled': self.stats['clip_enabled'],
            'clip_model': self.clip_model_name if self.use_clip else None
        }


class StructuralSimilarity:
    """
    Structural feature similarity scoring.

    Compares images based on low-level features (color, brightness, layout)
    when CLIP is unavailable or for specialized queries.

    Usage:
        scorer = StructuralSimilarity()
        similarity = scorer.compare(features1, features2)
        # Returns: 0.0-1.0 similarity score
    """

    def __init__(
        self,
        color_weight: float = 0.4,
        brightness_weight: float = 0.2,
        layout_weight: float = 0.2,
        edge_weight: float = 0.2
    ):
        """
        Initialize structural similarity scorer.

        Args:
            color_weight: Weight for color similarity (mean RGB)
            brightness_weight: Weight for brightness/contrast
            layout_weight: Weight for aspect ratio
            edge_weight: Weight for edge density
        """
        self.color_weight = color_weight
        self.brightness_weight = brightness_weight
        self.layout_weight = layout_weight
        self.edge_weight = edge_weight

        # Normalize weights
        total = color_weight + brightness_weight + layout_weight + edge_weight
        self.color_weight /= total
        self.brightness_weight /= total
        self.layout_weight /= total
        self.edge_weight /= total

    def compare(
        self,
        features1: Dict[str, float],
        features2: Dict[str, float]
    ) -> float:
        """
        Compare two sets of structural features.

        Args:
            features1: First image's structural features
            features2: Second image's structural features

        Returns:
            Similarity score (0.0-1.0), higher = more similar
        """
        # Color similarity (Euclidean distance in RGB space)
        color_dist = np.sqrt(
            (features1['mean_r'] - features2['mean_r']) ** 2 +
            (features1['mean_g'] - features2['mean_g']) ** 2 +
            (features1['mean_b'] - features2['mean_b']) ** 2
        )
        # Normalize to 0-1 (max distance = sqrt(3))
        color_sim = 1.0 - (color_dist / np.sqrt(3.0))

        # Brightness similarity
        brightness_dist = abs(features1['brightness'] - features2['brightness'])
        brightness_sim = 1.0 - brightness_dist

        # Layout similarity (aspect ratio)
        aspect_dist = abs(features1['aspect_ratio'] - features2['aspect_ratio'])
        # Normalize (aspect ratios typically 0.5-2.0)
        layout_sim = 1.0 - min(aspect_dist / 1.5, 1.0)

        # Edge density similarity
        edge_dist = abs(features1['edge_density'] - features2['edge_density'])
        edge_sim = 1.0 - min(edge_dist, 1.0)

        # Weighted combination
        total_sim = (
            self.color_weight * color_sim +
            self.brightness_weight * brightness_sim +
            self.layout_weight * layout_sim +
            self.edge_weight * edge_sim
        )

        return float(np.clip(total_sim, 0.0, 1.0))


# Convenience function
async def encode_image(
    image: np.ndarray,
    use_clip: bool = True,
    caption: Optional[str] = None
) -> ImageEmbeddings:
    """
    Convenience function for encoding images.

    Args:
        image: RGB image array
        use_clip: Enable CLIP encoding
        caption: Optional caption

    Returns:
        ImageEmbeddings
    """
    encoder = MultimodalEncoder(use_clip=use_clip)
    return await encoder.encode_image(image, caption=caption)
