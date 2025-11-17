"""
HoloLoom Multi-Modal Module
============================
Multi-modal embeddings and cross-modal search.

This module extends HoloLoom's text-only embeddings to support:
- Images (CLIP-based vision-language alignment)
- Audio (Whisper transcription + acoustic features)
- Video (frame extraction + audio processing)
- Cross-modal fusion and retrieval

All modalities share a unified 768D embedding space, enabling
seamless cross-modal operations.

Example Usage:
    # Image encoding
    from HoloLoom.multimodal import ImageEncoder
    encoder = ImageEncoder()
    embedding = await encoder.encode("cat.jpg")

    # Cross-modal search
    from HoloLoom.multimodal import MultiModalSearch, Modality
    search = MultiModalSearch()
    results = await search.search(
        query="cats playing",
        query_modality=Modality.TEXT,
        target_modalities=[Modality.IMAGE, Modality.VIDEO]
    )

    # Multi-modal fusion
    from HoloLoom.multimodal import MultiModalFuser
    fuser = MultiModalFuser(method="late")
    fused = fuser.fuse_embeddings({
        Modality.TEXT: text_embedding,
        Modality.IMAGE: image_embedding
    })
"""

# Base types and protocols
from HoloLoom.multimodal.base import (
    Modality,
    ModalEmbedding,
    MultiModalEmbedding,
    ModalEncoder,
    ImageEncoderConfig,
    AudioEncoderConfig,
    VideoEncoderConfig,
    FusionConfig,
    validate_embedding_dim,
    normalize_embedding
)

# Encoders
from HoloLoom.multimodal.image_encoder import (
    ImageEncoder,
    create_image_encoder
)

from HoloLoom.multimodal.audio_encoder import (
    AudioEncoder,
    create_audio_encoder
)

from HoloLoom.multimodal.video_encoder import (
    VideoEncoder,
    create_video_encoder
)

# Fusion
from HoloLoom.multimodal.fusion import (
    LateFusion,
    EarlyFusion,
    HybridFusion,
    MultiModalFuser,
    create_fuser
)

# Search
from HoloLoom.multimodal.search import (
    SearchResult,
    SearchResults,
    MultiModalSearch,
    create_search_engine
)

# Version
__version__ = "3.0.0"

# Public API
__all__ = [
    # Base
    "Modality",
    "ModalEmbedding",
    "MultiModalEmbedding",
    "ModalEncoder",
    "ImageEncoderConfig",
    "AudioEncoderConfig",
    "VideoEncoderConfig",
    "FusionConfig",
    "validate_embedding_dim",
    "normalize_embedding",

    # Encoders
    "ImageEncoder",
    "AudioEncoder",
    "VideoEncoder",
    "create_image_encoder",
    "create_audio_encoder",
    "create_video_encoder",

    # Fusion
    "LateFusion",
    "EarlyFusion",
    "HybridFusion",
    "MultiModalFuser",
    "create_fuser",

    # Search
    "SearchResult",
    "SearchResults",
    "MultiModalSearch",
    "create_search_engine",
]
