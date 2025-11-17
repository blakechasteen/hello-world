"""
HoloLoom Video Encoder
======================
Frame extraction and temporal encoding for video content.

This encoder processes videos by:
1. Extracting keyframes at configurable intervals
2. Detecting scene changes
3. Encoding visual content (via ImageEncoder)
4. Encoding audio track (via AudioEncoder)
5. Creating temporal embeddings

Features:
- Frame extraction at configurable FPS
- Scene change detection
- Audio track processing
- Temporal pooling strategies
- Support for MP4, AVI, MOV formats

Architecture:
- Input: Video file path
- Process:
  1. Extract frames (1 FPS default)
  2. Encode each frame → 768D (ImageEncoder)
  3. Extract audio → 768D (AudioEncoder)
  4. Temporal pooling → single 768D embedding
- Output: ModalEmbedding with 768D vector + metadata

The temporal pooling aggregates frame embeddings using
strategies like mean, max, or attention-weighted averaging.
"""

import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

# Import from base module
from HoloLoom.multimodal.base import (
    Modality,
    ModalEmbedding,
    VideoEncoderConfig,
    validate_embedding_dim,
    normalize_embedding
)

logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallback
try:
    import cv2
    _HAVE_CV2 = True
except ImportError:
    cv2 = None
    _HAVE_CV2 = False
    warnings.warn("cv2 not available. Install with: pip install opencv-python")

try:
    from scenedetect import detect, ContentDetector
    _HAVE_SCENEDETECT = True
except ImportError:
    ContentDetector = None
    _HAVE_SCENEDETECT = False
    warnings.warn("scenedetect not available. Install with: pip install scenedetect[opencv]")

# Import other encoders for video components
try:
    from HoloLoom.multimodal.image_encoder import ImageEncoder
    _HAVE_IMAGE_ENCODER = True
except ImportError:
    ImageEncoder = None
    _HAVE_IMAGE_ENCODER = False
    warnings.warn("ImageEncoder not available")

try:
    from HoloLoom.multimodal.audio_encoder import AudioEncoder
    _HAVE_AUDIO_ENCODER = True
except ImportError:
    AudioEncoder = None
    _HAVE_AUDIO_ENCODER = False
    warnings.warn("AudioEncoder not available")


# ============================================================================
# Video Encoder Implementation
# ============================================================================

class VideoEncoder:
    """
    Video encoder with frame and audio processing.

    Encodes videos into 768D embeddings that capture:
    1. Visual content (via frame embeddings)
    2. Audio content (via audio track embedding)
    3. Temporal dynamics (via scene changes and pooling)

    This enables video search and retrieval based on
    visual, audio, or combined queries.

    Technical Details:
    - Extracts frames at configurable rate (1 FPS default)
    - Scene detection using adaptive thresholding
    - Temporal pooling: mean, max, or attention-weighted
    - Audio: 30% weight, Visual: 70% weight (configurable)

    Example:
        encoder = VideoEncoder()
        embedding = await encoder.encode("video.mp4")
        # Returns ModalEmbedding with 768D vector + metadata
    """

    def __init__(self, config: Optional[VideoEncoderConfig] = None):
        """
        Initialize video encoder.

        Args:
            config: Optional configuration (uses defaults if None)
        """
        self.config = config or VideoEncoderConfig()

        # Initialize component encoders
        self._image_encoder = None
        self._audio_encoder = None

        if _HAVE_IMAGE_ENCODER:
            try:
                self._image_encoder = ImageEncoder()
            except Exception as e:
                logger.error(f"Failed to initialize image encoder: {e}")

        if _HAVE_AUDIO_ENCODER and self.config.enable_audio:
            try:
                self._audio_encoder = AudioEncoder()
            except Exception as e:
                logger.error(f"Failed to initialize audio encoder: {e}")

    @property
    def modality(self) -> Modality:
        """Return modality type."""
        return Modality.VIDEO

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return 768

    def _extract_frames(
        self,
        video_path: Union[str, Path]
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Extract frames from video at configured rate.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (frames, metadata)
            - frames: List of frame arrays (RGB)
            - metadata: Video properties (fps, duration, resolution)

        Raises:
            RuntimeError: If CV2 not available
            FileNotFoundError: If video file not found
        """
        if not _HAVE_CV2:
            raise RuntimeError("cv2 required for video processing: pip install opencv-python")

        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")

        try:
            # Open video
            cap = cv2.VideoCapture(str(path))

            # Get properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0

            # Calculate frame sampling interval
            frame_interval = int(fps / self.config.frame_rate) if fps > 0 else 1
            frame_interval = max(1, frame_interval)

            # Extract frames
            frames = []
            frame_idx = 0

            while len(frames) < self.config.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample frames at interval
                if frame_idx % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

                frame_idx += 1

            cap.release()

            metadata = {
                "fps": fps,
                "duration": duration,
                "total_frames": total_frames,
                "extracted_frames": len(frames),
                "resolution": (width, height)
            }

            logger.info(
                f"Extracted {len(frames)} frames from {duration:.2f}s video "
                f"({fps:.2f} FPS)"
            )

            return frames, metadata

        except Exception as e:
            logger.error(f"Failed to extract frames from {path}: {e}")
            raise

    def _detect_scenes(
        self,
        video_path: Union[str, Path]
    ) -> List[Tuple[float, float]]:
        """
        Detect scene changes in video.

        Args:
            video_path: Path to video file

        Returns:
            List of (start_time, end_time) tuples for each scene

        Note:
            Falls back to uniform segmentation if scenedetect unavailable
        """
        if not self.config.enable_scene_detection:
            return []

        if not _HAVE_SCENEDETECT:
            logger.warning("scenedetect unavailable - skipping scene detection")
            return []

        try:
            # Detect scenes
            scene_list = detect(str(video_path), ContentDetector())

            # Convert to time tuples
            scenes = [
                (scene[0].get_seconds(), scene[1].get_seconds())
                for scene in scene_list
            ]

            logger.info(f"Detected {len(scenes)} scenes")
            return scenes

        except Exception as e:
            logger.warning(f"Scene detection failed: {e}")
            return []

    def _extract_audio(
        self,
        video_path: Union[str, Path]
    ) -> Optional[str]:
        """
        Extract audio track from video to temporary file.

        Args:
            video_path: Path to video file

        Returns:
            Path to extracted audio file (WAV) or None if failed

        Note:
            Uses ffmpeg via subprocess. Requires ffmpeg installed.
        """
        if not self.config.enable_audio:
            return None

        try:
            import subprocess
            import tempfile

            # Create temporary file
            temp_audio = tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            )
            temp_audio.close()

            # Extract audio with ffmpeg
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM codec
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",  # Mono
                "-y",  # Overwrite
                temp_audio.name
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=60
            )

            if result.returncode == 0:
                logger.info(f"Extracted audio to {temp_audio.name}")
                return temp_audio.name
            else:
                logger.warning(f"Audio extraction failed: {result.stderr}")
                return None

        except Exception as e:
            logger.warning(f"Audio extraction failed: {e}")
            return None

    def _temporal_pool(
        self,
        frame_embeddings: List[np.ndarray],
        strategy: str = "mean"
    ) -> np.ndarray:
        """
        Pool frame embeddings into single temporal embedding.

        Args:
            frame_embeddings: List of 768D frame embeddings
            strategy: Pooling strategy ("mean", "max", "attention")

        Returns:
            Single 768D embedding
        """
        if not frame_embeddings:
            return np.zeros(768)

        embeddings = np.array(frame_embeddings)

        if strategy == "mean":
            # Average pooling
            pooled = np.mean(embeddings, axis=0)

        elif strategy == "max":
            # Max pooling (element-wise max)
            pooled = np.max(embeddings, axis=0)

        elif strategy == "attention":
            # Attention-weighted pooling
            # Compute attention as L2 norm (saliency)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            weights = norms / (np.sum(norms) + 1e-9)
            pooled = np.sum(embeddings * weights, axis=0)

        else:
            # Default to mean
            pooled = np.mean(embeddings, axis=0)

        # Normalize
        pooled = pooled / (np.linalg.norm(pooled) + 1e-9)

        return pooled

    async def encode(
        self,
        input_data: Union[str, Path],
        **kwargs
    ) -> ModalEmbedding:
        """
        Encode single video to 768D embedding.

        Args:
            input_data: Video file path
            **kwargs: Additional options
                - pooling: Temporal pooling strategy ("mean", "max", "attention")
                - visual_weight: Weight for visual features (default 0.7)
                - audio_weight: Weight for audio features (default 0.3)

        Returns:
            ModalEmbedding with 768D vector + metadata

        Raises:
            RuntimeError: If required dependencies unavailable
        """
        pooling = kwargs.get("pooling", "mean")
        visual_weight = kwargs.get("visual_weight", 0.7)
        audio_weight = kwargs.get("audio_weight", 0.3)

        video_path = Path(input_data)

        # Extract frames
        frames, video_metadata = self._extract_frames(video_path)

        # Detect scenes (optional)
        scenes = self._detect_scenes(video_path)

        # Encode frames
        frame_embeddings = []
        if self._image_encoder is not None and frames:
            logger.info(f"Encoding {len(frames)} frames...")
            for i, frame in enumerate(frames):
                try:
                    # Convert numpy array to PIL Image
                    from PIL import Image
                    pil_image = Image.fromarray(frame)

                    # Encode frame
                    frame_emb = await self._image_encoder.encode(
                        pil_image,
                        path=f"{video_path}_frame_{i}"
                    )
                    frame_embeddings.append(np.array(frame_emb.embedding))

                except Exception as e:
                    logger.warning(f"Failed to encode frame {i}: {e}")
                    continue

        # Temporal pooling
        visual_embedding = self._temporal_pool(frame_embeddings, strategy=pooling)

        # Extract and encode audio
        audio_embedding = np.zeros(768)
        audio_path = None
        transcription = ""

        if self._audio_encoder is not None and self.config.enable_audio:
            audio_path = self._extract_audio(video_path)
            if audio_path:
                try:
                    audio_emb = await self._audio_encoder.encode(audio_path)
                    audio_embedding = np.array(audio_emb.embedding)
                    transcription = audio_emb.metadata.get("transcription", "")
                except Exception as e:
                    logger.warning(f"Failed to encode audio: {e}")

                # Clean up temporary audio file
                try:
                    Path(audio_path).unlink()
                except Exception:
                    pass

        # Combine visual and audio
        combined_embedding = (
            visual_weight * visual_embedding +
            audio_weight * audio_embedding
        )

        # Normalize
        combined_embedding = combined_embedding / (
            np.linalg.norm(combined_embedding) + 1e-9
        )

        # Validate dimension
        validate_embedding_dim(combined_embedding.tolist(), 768)

        # Build metadata
        metadata = {
            "file": str(video_path),
            "duration": video_metadata.get("duration", 0),
            "fps": video_metadata.get("fps", 0),
            "resolution": video_metadata.get("resolution", (0, 0)),
            "total_frames": video_metadata.get("total_frames", 0),
            "extracted_frames": video_metadata.get("extracted_frames", 0),
            "scenes": len(scenes),
            "transcription": transcription,
            "pooling_strategy": pooling
        }

        confidence = 1.0 if frame_embeddings else 0.5

        return ModalEmbedding(
            embedding=combined_embedding.tolist(),
            modality=Modality.VIDEO,
            metadata=metadata,
            confidence=confidence
        )

    async def encode_batch(
        self,
        inputs: List[Union[str, Path]],
        **kwargs
    ) -> List[ModalEmbedding]:
        """
        Encode multiple videos efficiently.

        Args:
            inputs: List of video paths
            **kwargs: Additional options

        Returns:
            List of ModalEmbeddings
        """
        results = []
        for input_data in inputs:
            try:
                embedding = await self.encode(input_data, **kwargs)
                results.append(embedding)
            except Exception as e:
                logger.error(f"Failed to encode {input_data}: {e}")
                # Continue with other videos
                continue

        return results


# ============================================================================
# Factory Function
# ============================================================================

def create_video_encoder(
    frame_rate: float = 1.0,
    enable_scene_detection: bool = True,
    enable_audio: bool = True,
    device: str = "cpu",
    **kwargs
) -> VideoEncoder:
    """
    Factory function to create video encoder.

    Args:
        frame_rate: Frames per second to extract (default 1.0)
        enable_scene_detection: Enable scene detection
        enable_audio: Enable audio track processing
        device: "cpu" or "cuda"
        **kwargs: Additional config options

    Returns:
        VideoEncoder instance
    """
    config = VideoEncoderConfig(
        frame_rate=frame_rate,
        enable_scene_detection=enable_scene_detection,
        enable_audio=enable_audio,
        device=device,
        **kwargs
    )
    return VideoEncoder(config)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("=" * 80)
        print("Video Encoder Demo")
        print("=" * 80 + "\n")

        # Create encoder
        encoder = create_video_encoder(
            frame_rate=1.0,
            enable_scene_detection=True,
            enable_audio=True
        )

        print(f"Modality: {encoder.modality.value}")
        print(f"Embedding dimension: {encoder.embedding_dim}")
        print(f"CV2 available: {_HAVE_CV2}")
        print(f"Scene detection available: {_HAVE_SCENEDETECT}")
        print(f"Image encoder available: {_HAVE_IMAGE_ENCODER}")
        print(f"Audio encoder available: {_HAVE_AUDIO_ENCODER}")

        if not _HAVE_CV2:
            print("\nCV2 not available - install with: pip install opencv-python")
            return

        # Would encode real video:
        # embedding = await encoder.encode("path/to/video.mp4")
        # print(f"\nEmbedding shape: {len(embedding.embedding)}")
        # print(f"Duration: {embedding.metadata.get('duration'):.2f}s")
        # print(f"Frames: {embedding.metadata.get('extracted_frames')}")
        # print(f"Scenes: {embedding.metadata.get('scenes')}")
        # print(f"Transcription: {embedding.metadata.get('transcription')}")

        print("\n✓ Video encoder ready!")
        print("Usage: embedding = await encoder.encode('video.mp4')")

    asyncio.run(demo())
