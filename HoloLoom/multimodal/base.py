"""
HoloLoom Multi-Modal Base
==========================
Core protocols and types for multi-modal embeddings.

This module defines the foundation for multi-modal understanding:
- Modality enumeration (TEXT, IMAGE, AUDIO, VIDEO)
- ModalEncoder protocol for implementing encoders
- Unified embedding output format
- Configuration dataclasses

Philosophy:
All modalities are projected into a shared 768-dimensional embedding space,
enabling cross-modal retrieval and reasoning.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, List, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path

# Import from shared types layer
from HoloLoom.Documentation.types import Vector


# ============================================================================
# Modality Enumeration
# ============================================================================

class Modality(Enum):
    """
    Supported modality types.

    Each modality has unique characteristics:
    - TEXT: Sequential tokens, semantic embeddings
    - IMAGE: Spatial features, visual-semantic alignment (CLIP)
    - AUDIO: Temporal waveforms, speech-to-text
    - VIDEO: Spatiotemporal, combines image+audio+temporal
    """
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


# ============================================================================
# Embedding Output Format
# ============================================================================

@dataclass
class ModalEmbedding:
    """
    Unified embedding output from any modality encoder.

    All encoders produce embeddings in the same 768D space,
    enabling seamless cross-modal operations.

    Attributes:
        embedding: 768-dimensional vector
        modality: Source modality (TEXT, IMAGE, AUDIO, VIDEO)
        metadata: Additional information (file path, extracted text, etc.)
        confidence: Encoding confidence (0.0 to 1.0)
    """
    embedding: Vector                               # 768D vector
    modality: Modality                              # Source modality
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0                         # Encoding confidence


@dataclass
class MultiModalEmbedding:
    """
    Combined embedding from multiple modalities.

    Used when processing content with multiple modalities
    (e.g., video = visual + audio, document with images).

    Attributes:
        embeddings: Dict mapping modality to embedding
        fused_embedding: Optional unified embedding across modalities
        metadata: Fusion metadata
    """
    embeddings: Dict[Modality, Vector] = field(default_factory=dict)
    fused_embedding: Optional[Vector] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Encoder Protocol
# ============================================================================

class ModalEncoder(Protocol):
    """
    Protocol for modality-specific encoders.

    All encoders must implement:
    - encode(): Convert raw input to embedding
    - encode_batch(): Batch encoding for efficiency
    - modality property: Return supported modality

    This protocol enables swapping implementations without changing
    orchestrator code (following HoloLoom's protocol-based design).
    """

    @property
    def modality(self) -> Modality:
        """Return the modality this encoder handles."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension (should be 768 for cross-modal compatibility)."""
        ...

    async def encode(
        self,
        input_data: Union[str, Path, np.ndarray],
        **kwargs
    ) -> ModalEmbedding:
        """
        Encode single input to embedding.

        Args:
            input_data: Raw input (file path, array, etc.)
            **kwargs: Encoder-specific parameters

        Returns:
            ModalEmbedding with 768D vector
        """
        ...

    async def encode_batch(
        self,
        inputs: List[Union[str, Path, np.ndarray]],
        **kwargs
    ) -> List[ModalEmbedding]:
        """
        Encode multiple inputs efficiently.

        Args:
            inputs: List of raw inputs
            **kwargs: Encoder-specific parameters

        Returns:
            List of ModalEmbeddings
        """
        ...


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class ImageEncoderConfig:
    """Configuration for image encoder."""
    model_name: str = "openai/clip-vit-base-patch32"  # CLIP model
    enable_ocr: bool = True                            # Extract text from images
    ocr_lang: str = "eng"                              # OCR language
    cache_dir: Optional[str] = None                    # Model cache directory
    device: str = "cpu"                                # "cpu" or "cuda"


@dataclass
class AudioEncoderConfig:
    """Configuration for audio encoder."""
    model_name: str = "base"                           # Whisper model size
    enable_diarization: bool = False                   # Speaker diarization
    language: Optional[str] = None                     # Auto-detect if None
    cache_dir: Optional[str] = None
    device: str = "cpu"


@dataclass
class VideoEncoderConfig:
    """Configuration for video encoder."""
    frame_rate: float = 1.0                            # Frames per second to extract
    enable_scene_detection: bool = True                # Detect scene changes
    enable_audio: bool = True                          # Process audio track
    max_frames: int = 300                              # Maximum frames to process
    cache_dir: Optional[str] = None
    device: str = "cpu"


@dataclass
class FusionConfig:
    """Configuration for cross-modal fusion."""
    method: str = "late"                               # "early", "late", "hybrid"
    weights: Dict[Modality, float] = field(default_factory=lambda: {
        Modality.TEXT: 0.4,
        Modality.IMAGE: 0.3,
        Modality.AUDIO: 0.2,
        Modality.VIDEO: 0.1
    })
    normalize: bool = True                             # L2 normalize fused embeddings


# ============================================================================
# Utility Functions
# ============================================================================

def validate_embedding_dim(embedding: Vector, expected_dim: int = 768) -> bool:
    """
    Validate embedding dimension.

    Args:
        embedding: Vector to validate
        expected_dim: Expected dimension (default 768)

    Returns:
        True if valid

    Raises:
        ValueError: If dimension mismatch
    """
    if len(embedding) != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}"
        )
    return True


def normalize_embedding(embedding: Vector) -> Vector:
    """
    L2-normalize embedding vector.

    Args:
        embedding: Vector to normalize

    Returns:
        Normalized vector
    """
    arr = np.array(embedding)
    norm = np.linalg.norm(arr)
    if norm < 1e-9:
        return embedding
    return (arr / norm).tolist()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=== HoloLoom Multi-Modal Base ===\n")

    # Example: Create embedding
    embedding = ModalEmbedding(
        embedding=[0.1] * 768,
        modality=Modality.IMAGE,
        metadata={"file": "example.jpg"},
        confidence=0.95
    )

    print(f"Modality: {embedding.modality.value}")
    print(f"Dimension: {len(embedding.embedding)}")
    print(f"Confidence: {embedding.confidence}")
    print(f"Metadata: {embedding.metadata}")

    # Example: Multi-modal embedding
    multi = MultiModalEmbedding(
        embeddings={
            Modality.IMAGE: [0.1] * 768,
            Modality.TEXT: [0.2] * 768
        },
        fused_embedding=[0.15] * 768,
        metadata={"source": "document_with_images.pdf"}
    )

    print(f"\nMulti-modal: {list(multi.embeddings.keys())}")
    print(f"Has fused: {multi.fused_embedding is not None}")

    print("\n✓ Base module loaded successfully!")
