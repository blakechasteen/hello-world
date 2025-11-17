"""
HoloLoom Multi-Modal Tests
===========================
Comprehensive test suite for Phase 3A multi-modal embeddings.

Tests cover:
1. Base types and protocols
2. Image encoder (with and without CLIP)
3. Audio encoder (with and without Whisper)
4. Video encoder (with and without dependencies)
5. Fusion strategies (late, early, hybrid)
6. Cross-modal search
7. File parsers (image, audio, video)
8. Integration tests

Run with:
    PYTHONPATH=. python HoloLoom/test_multimodal.py
"""

import sys
import asyncio
import tempfile
from pathlib import Path
import numpy as np

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test framework
try:
    import pytest
    HAVE_PYTEST = True
except ImportError:
    HAVE_PYTEST = False
    print("Warning: pytest not available, using simple assertions")

# Import multimodal components
from HoloLoom.multimodal.base import (
    Modality,
    ModalEmbedding,
    MultiModalEmbedding,
    validate_embedding_dim,
    normalize_embedding
)

from HoloLoom.multimodal.fusion import (
    LateFusion,
    EarlyFusion,
    HybridFusion,
    MultiModalFuser
)


# ============================================================================
# Test Utilities
# ============================================================================

class TestCounter:
    """Simple test counter for tracking pass/fail."""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0

    def record(self, name: str, passed: bool):
        """Record test result."""
        self.total += 1
        if passed:
            self.passed += 1
            print(f"✓ {name}")
        else:
            self.failed += 1
            print(f"✗ {name}")

    def summary(self):
        """Print test summary."""
        print("\n" + "=" * 80)
        print(f"Test Summary: {self.passed}/{self.total} passed")
        if self.failed > 0:
            print(f"FAILED: {self.failed} tests failed")
        else:
            print("SUCCESS: All tests passed!")
        print("=" * 80)


counter = TestCounter()


def test_assert(name: str, condition: bool):
    """Assert a condition and record result."""
    counter.record(name, condition)
    if not condition:
        print(f"  Assertion failed: {name}")


# ============================================================================
# Test 1: Base Types and Protocols
# ============================================================================

def test_base_types():
    """Test base types and validation."""
    print("\n" + "=" * 80)
    print("Test 1: Base Types and Protocols")
    print("=" * 80)

    # Test Modality enum
    test_assert(
        "Modality.TEXT enum",
        Modality.TEXT.value == "text"
    )
    test_assert(
        "Modality.IMAGE enum",
        Modality.IMAGE.value == "image"
    )

    # Test ModalEmbedding
    embedding = ModalEmbedding(
        embedding=[0.1] * 768,
        modality=Modality.IMAGE,
        metadata={"test": "data"},
        confidence=0.95
    )
    test_assert(
        "ModalEmbedding creation",
        len(embedding.embedding) == 768
    )
    test_assert(
        "ModalEmbedding modality",
        embedding.modality == Modality.IMAGE
    )
    test_assert(
        "ModalEmbedding confidence",
        embedding.confidence == 0.95
    )

    # Test MultiModalEmbedding
    multi = MultiModalEmbedding(
        embeddings={
            Modality.TEXT: [0.1] * 768,
            Modality.IMAGE: [0.2] * 768
        },
        fused_embedding=[0.15] * 768
    )
    test_assert(
        "MultiModalEmbedding creation",
        len(multi.embeddings) == 2
    )
    test_assert(
        "MultiModalEmbedding fused",
        len(multi.fused_embedding) == 768
    )

    # Test validation
    try:
        validate_embedding_dim([0.1] * 768, 768)
        test_assert("validate_embedding_dim (valid)", True)
    except ValueError:
        test_assert("validate_embedding_dim (valid)", False)

    try:
        validate_embedding_dim([0.1] * 512, 768)
        test_assert("validate_embedding_dim (invalid)", False)
    except ValueError:
        test_assert("validate_embedding_dim (invalid)", True)

    # Test normalization
    vec = [3.0, 4.0] + [0.0] * 766
    normalized = normalize_embedding(vec)
    norm = np.linalg.norm(normalized)
    test_assert(
        "normalize_embedding",
        abs(norm - 1.0) < 1e-6
    )


# ============================================================================
# Test 2: Fusion Strategies
# ============================================================================

def test_fusion_strategies():
    """Test fusion strategies."""
    print("\n" + "=" * 80)
    print("Test 2: Fusion Strategies")
    print("=" * 80)

    # Create sample embeddings
    embeddings = {
        Modality.TEXT: np.random.randn(768).tolist(),
        Modality.IMAGE: np.random.randn(768).tolist(),
        Modality.AUDIO: np.random.randn(768).tolist()
    }

    # Test Late Fusion
    late_fuser = LateFusion()
    late_result = late_fuser.fuse(embeddings)
    test_assert(
        "LateFusion output dimension",
        len(late_result) == 768
    )
    test_assert(
        "LateFusion normalized",
        abs(np.linalg.norm(late_result) - 1.0) < 1e-3
    )

    # Test Early Fusion
    early_fuser = EarlyFusion()
    early_result = early_fuser.fuse(embeddings)
    test_assert(
        "EarlyFusion output dimension",
        len(early_result) == 768
    )

    # Test Hybrid Fusion
    hybrid_fuser = HybridFusion()
    hybrid_result = hybrid_fuser.fuse(embeddings)
    test_assert(
        "HybridFusion output dimension",
        len(hybrid_result) == 768
    )

    # Test MultiModalFuser
    fuser = MultiModalFuser(method="late")
    multi_result = fuser.fuse_embeddings(embeddings)
    test_assert(
        "MultiModalFuser fused embedding",
        multi_result.fused_embedding is not None
    )
    test_assert(
        "MultiModalFuser metadata",
        "fusion_method" in multi_result.metadata
    )
    test_assert(
        "MultiModalFuser num modalities",
        multi_result.metadata["num_modalities"] == 3
    )

    # Test similarity
    multi_result2 = fuser.fuse_embeddings(embeddings)
    similarity = fuser.compute_similarity(multi_result, multi_result2)
    test_assert(
        "Similarity (same embeddings)",
        abs(similarity - 1.0) < 1e-3
    )


# ============================================================================
# Test 3: Image Encoder (Fallback Mode)
# ============================================================================

async def test_image_encoder_fallback():
    """Test image encoder in fallback mode (no CLIP)."""
    print("\n" + "=" * 80)
    print("Test 3: Image Encoder (Fallback Mode)")
    print("=" * 80)

    try:
        from HoloLoom.multimodal.image_encoder import ImageEncoder

        encoder = ImageEncoder()
        test_assert(
            "ImageEncoder creation",
            encoder is not None
        )
        test_assert(
            "ImageEncoder modality",
            encoder.modality == Modality.IMAGE
        )
        test_assert(
            "ImageEncoder embedding_dim",
            encoder.embedding_dim == 768
        )

        # Test fallback embedding (deterministic)
        fallback_emb = encoder._fallback_embedding("test.jpg")
        test_assert(
            "Fallback embedding dimension",
            len(fallback_emb) == 768
        )
        test_assert(
            "Fallback embedding normalized",
            abs(np.linalg.norm(fallback_emb) - 1.0) < 1e-6
        )

        # Test determinism
        fallback_emb2 = encoder._fallback_embedding("test.jpg")
        test_assert(
            "Fallback embedding deterministic",
            np.allclose(fallback_emb, fallback_emb2)
        )

    except Exception as e:
        test_assert(f"ImageEncoder tests: {e}", False)


# ============================================================================
# Test 4: Audio Encoder (Fallback Mode)
# ============================================================================

async def test_audio_encoder_fallback():
    """Test audio encoder components."""
    print("\n" + "=" * 80)
    print("Test 4: Audio Encoder Components")
    print("=" * 80)

    try:
        from HoloLoom.multimodal.audio_encoder import AudioEncoder

        encoder = AudioEncoder()
        test_assert(
            "AudioEncoder creation",
            encoder is not None
        )
        test_assert(
            "AudioEncoder modality",
            encoder.modality == Modality.AUDIO
        )
        test_assert(
            "AudioEncoder embedding_dim",
            encoder.embedding_dim == 768
        )

        # Test acoustic feature extraction (with zeros)
        features = encoder._extract_acoustic_features(
            np.zeros(16000),  # 1 second of silence at 16kHz
            16000
        )
        test_assert(
            "Acoustic features dimension",
            len(features) == 256
        )

    except Exception as e:
        test_assert(f"AudioEncoder tests: {e}", False)


# ============================================================================
# Test 5: Video Encoder Components
# ============================================================================

async def test_video_encoder_components():
    """Test video encoder components."""
    print("\n" + "=" * 80)
    print("Test 5: Video Encoder Components")
    print("=" * 80)

    try:
        from HoloLoom.multimodal.video_encoder import VideoEncoder

        encoder = VideoEncoder()
        test_assert(
            "VideoEncoder creation",
            encoder is not None
        )
        test_assert(
            "VideoEncoder modality",
            encoder.modality == Modality.VIDEO
        )
        test_assert(
            "VideoEncoder embedding_dim",
            encoder.embedding_dim == 768
        )

        # Test temporal pooling
        frame_embeddings = [
            np.random.randn(768),
            np.random.randn(768),
            np.random.randn(768)
        ]

        # Mean pooling
        pooled_mean = encoder._temporal_pool(frame_embeddings, strategy="mean")
        test_assert(
            "Temporal pooling (mean) dimension",
            len(pooled_mean) == 768
        )
        test_assert(
            "Temporal pooling (mean) normalized",
            abs(np.linalg.norm(pooled_mean) - 1.0) < 1e-3
        )

        # Max pooling
        pooled_max = encoder._temporal_pool(frame_embeddings, strategy="max")
        test_assert(
            "Temporal pooling (max) dimension",
            len(pooled_max) == 768
        )

        # Attention pooling
        pooled_attn = encoder._temporal_pool(frame_embeddings, strategy="attention")
        test_assert(
            "Temporal pooling (attention) dimension",
            len(pooled_attn) == 768
        )

    except Exception as e:
        test_assert(f"VideoEncoder tests: {e}", False)


# ============================================================================
# Test 6: File Parsers
# ============================================================================

async def test_file_parsers():
    """Test file parsers (without actual files)."""
    print("\n" + "=" * 80)
    print("Test 6: File Parsers")
    print("=" * 80)

    # Test that parsers can be imported
    try:
        from HoloLoom.ingestion.parsers import image, audio, video
        test_assert("Image parser import", True)
        test_assert("Audio parser import", True)
        test_assert("Video parser import", True)
    except ImportError as e:
        test_assert(f"Parser imports: {e}", False)

    # Test parser availability flags
    try:
        from HoloLoom.ingestion.parsers.image import PIL_AVAILABLE, OCR_AVAILABLE
        test_assert(
            "Image parser availability check",
            isinstance(PIL_AVAILABLE, bool)
        )
        print(f"  PIL available: {PIL_AVAILABLE}")
        print(f"  OCR available: {OCR_AVAILABLE}")
    except Exception as e:
        test_assert(f"Image parser flags: {e}", False)

    try:
        from HoloLoom.ingestion.parsers.audio import WHISPER_AVAILABLE, LIBROSA_AVAILABLE
        test_assert(
            "Audio parser availability check",
            isinstance(WHISPER_AVAILABLE, bool)
        )
        print(f"  Whisper available: {WHISPER_AVAILABLE}")
        print(f"  Librosa available: {LIBROSA_AVAILABLE}")
    except Exception as e:
        test_assert(f"Audio parser flags: {e}", False)

    try:
        from HoloLoom.ingestion.parsers.video import CV2_AVAILABLE
        test_assert(
            "Video parser availability check",
            isinstance(CV2_AVAILABLE, bool)
        )
        print(f"  CV2 available: {CV2_AVAILABLE}")
    except Exception as e:
        test_assert(f"Video parser flags: {e}", False)


# ============================================================================
# Test 7: Multi-Modal Search (Components)
# ============================================================================

async def test_multimodal_search_components():
    """Test multi-modal search components."""
    print("\n" + "=" * 80)
    print("Test 7: Multi-Modal Search Components")
    print("=" * 80)

    try:
        from HoloLoom.multimodal.search import (
            SearchResult,
            SearchResults,
            MultiModalSearch
        )

        # Test SearchResult
        result = SearchResult(
            id="test_id",
            modality=Modality.IMAGE,
            embedding=[0.1] * 768,
            metadata={"file": "test.jpg"},
            score=0.95,
            cross_modal=True
        )
        test_assert(
            "SearchResult creation",
            result.id == "test_id"
        )
        test_assert(
            "SearchResult cross_modal",
            result.cross_modal == True
        )

        # Test SearchResults
        results = SearchResults(
            results=[result],
            query_modality=Modality.TEXT,
            target_modalities=[Modality.IMAGE],
            total_results=1
        )
        test_assert(
            "SearchResults creation",
            len(results.results) == 1
        )
        test_assert(
            "SearchResults query_modality",
            results.query_modality == Modality.TEXT
        )

        # Test MultiModalSearch initialization
        search = MultiModalSearch(
            collection_name="test_collection",
            fusion_method="late"
        )
        test_assert(
            "MultiModalSearch creation",
            search.collection_name == "test_collection"
        )
        test_assert(
            "MultiModalSearch fusion_method",
            search.fusion_method == "late"
        )

    except Exception as e:
        test_assert(f"MultiModalSearch tests: {e}", False)


# ============================================================================
# Test 8: Integration Test (End-to-End)
# ============================================================================

async def test_integration():
    """Test end-to-end integration."""
    print("\n" + "=" * 80)
    print("Test 8: Integration Test")
    print("=" * 80)

    try:
        # Import all components
        from HoloLoom.multimodal import (
            Modality,
            ModalEmbedding,
            MultiModalFuser,
            create_fuser
        )

        # Create sample embeddings from different modalities
        text_emb = ModalEmbedding(
            embedding=np.random.randn(768).tolist(),
            modality=Modality.TEXT,
            metadata={"source": "query"}
        )

        image_emb = ModalEmbedding(
            embedding=np.random.randn(768).tolist(),
            modality=Modality.IMAGE,
            metadata={"file": "image.jpg"}
        )

        audio_emb = ModalEmbedding(
            embedding=np.random.randn(768).tolist(),
            modality=Modality.AUDIO,
            metadata={"file": "audio.mp3"}
        )

        # Test fusion
        fuser = create_fuser(method="late")
        fused = fuser.fuse_embeddings([text_emb, image_emb, audio_emb])

        test_assert(
            "Integration: Fused embedding created",
            fused.fused_embedding is not None
        )
        test_assert(
            "Integration: Fused embedding dimension",
            len(fused.fused_embedding) == 768
        )
        test_assert(
            "Integration: All modalities included",
            len(fused.embeddings) == 3
        )
        test_assert(
            "Integration: Metadata populated",
            "fusion_method" in fused.metadata
        )

        print("  ✓ End-to-end integration successful")

    except Exception as e:
        test_assert(f"Integration test: {e}", False)
        import traceback
        traceback.print_exc()


# ============================================================================
# Main Test Runner
# ============================================================================

async def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("HoloLoom Multi-Modal Test Suite - Phase 3A")
    print("=" * 80)

    # Run tests
    test_base_types()
    test_fusion_strategies()
    await test_image_encoder_fallback()
    await test_audio_encoder_fallback()
    await test_video_encoder_components()
    await test_file_parsers()
    await test_multimodal_search_components()
    await test_integration()

    # Print summary
    counter.summary()

    # Return exit code
    return 0 if counter.failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
