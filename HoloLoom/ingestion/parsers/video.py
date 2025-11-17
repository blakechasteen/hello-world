"""
Video Parser - Extract Content from Video Files
================================================
Parse video files and extract frame descriptions, audio transcriptions, and metadata.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import tempfile

logger = logging.getLogger(__name__)

# Try imports with graceful fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    logger.warning("cv2 not installed. Video parsing unavailable.")
    CV2_AVAILABLE = False
    cv2 = None

try:
    from HoloLoom.ingestion.parsers.audio import parse_audio, WHISPER_AVAILABLE
    AUDIO_PARSER_AVAILABLE = True
except ImportError:
    logger.warning("Audio parser not available.")
    AUDIO_PARSER_AVAILABLE = False
    WHISPER_AVAILABLE = False


async def parse_video(
    file_path: str,
    extract_audio: bool = True,
    sample_frames: int = 10,
    whisper_model: str = "base"
) -> str:
    """
    Extract content and metadata from video file.

    Args:
        file_path: Path to video file
        extract_audio: Extract and transcribe audio track
        sample_frames: Number of frames to sample for description
        whisper_model: Whisper model for audio transcription

    Returns:
        Extracted text with video info, frame descriptions, and transcription

    Raises:
        RuntimeError: If CV2 not available
        FileNotFoundError: If video file not found
    """
    if not CV2_AVAILABLE:
        raise RuntimeError("Video parsing requires cv2: pip install opencv-python")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    text_parts = []

    try:
        # Open video
        cap = cv2.VideoCapture(str(path))

        # Get metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        # Add header
        text_parts.append(f"[Video: {path.name}]")
        text_parts.append(f"Duration: {duration:.2f} seconds")
        text_parts.append(f"Resolution: {width}x{height}")
        text_parts.append(f"FPS: {fps:.2f}")
        text_parts.append(f"Total Frames: {total_frames}")

        # Sample frames
        if sample_frames > 0:
            frame_interval = max(1, total_frames // sample_frames)
            sampled_frames = 0
            frame_idx = 0

            text_parts.append(f"\nSampled Frames ({sample_frames}):")

            while sampled_frames < sample_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    # Describe frame (basic)
                    timestamp = frame_idx / fps if fps > 0 else 0
                    text_parts.append(
                        f"  Frame {frame_idx} (t={timestamp:.2f}s): "
                        f"{width}x{height} frame"
                    )
                    sampled_frames += 1

                frame_idx += 1

        cap.release()

        # Extract and transcribe audio
        if extract_audio and AUDIO_PARSER_AVAILABLE and WHISPER_AVAILABLE:
            audio_path = await _extract_audio_track(file_path)
            if audio_path:
                try:
                    logger.info("Transcribing audio track...")
                    audio_text = await parse_audio(
                        audio_path,
                        model_name=whisper_model
                    )
                    text_parts.append("\n--- Audio Track ---")
                    text_parts.append(audio_text)
                except Exception as e:
                    logger.warning(f"Audio transcription failed: {e}")
                    text_parts.append("\n[Audio transcription failed]")
                finally:
                    # Clean up temp file
                    try:
                        Path(audio_path).unlink()
                    except Exception:
                        pass
            else:
                text_parts.append("\n[No audio track found]")
        elif extract_audio and not WHISPER_AVAILABLE:
            text_parts.append("\n[Whisper not available for audio transcription]")

        # Combine all parts
        full_text = "\n".join(text_parts)

        logger.info(
            f"Parsed video {path.name}: {duration:.2f}s, "
            f"{total_frames} frames, {len(full_text)} chars"
        )

        return full_text

    except Exception as e:
        logger.error(f"Failed to parse video {file_path}: {e}")
        raise


async def _extract_audio_track(video_path: str) -> Optional[str]:
    """
    Extract audio track from video to temporary WAV file.

    Args:
        video_path: Path to video file

    Returns:
        Path to extracted audio file or None if failed

    Note:
        Requires ffmpeg installed on system
    """
    try:
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

    except subprocess.TimeoutExpired:
        logger.error("Audio extraction timed out")
        return None
    except FileNotFoundError:
        logger.error("ffmpeg not found - install ffmpeg to extract audio")
        return None
    except Exception as e:
        logger.warning(f"Audio extraction failed: {e}")
        return None


async def get_video_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from video file.

    Args:
        file_path: Path to video file

    Returns:
        Dict with metadata (duration, fps, resolution, codec, etc.)
    """
    if not CV2_AVAILABLE:
        return {}

    path = Path(file_path)
    if not path.exists():
        return {}

    try:
        cap = cv2.VideoCapture(str(path))

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        # Get codec (fourcc)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        cap.release()

        metadata = {
            "duration": duration,
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "resolution": (width, height),
            "codec": codec,
            "format": path.suffix[1:].upper()
        }

        return metadata

    except Exception as e:
        logger.error(f"Failed to extract metadata from {file_path}: {e}")
        return {}


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("=" * 80)
        print("Video Parser Demo")
        print("=" * 80 + "\n")

        if not CV2_AVAILABLE:
            print("CV2 not available")
            print("Install with: pip install opencv-python")
            return

        print("Video parser is ready")
        print("Supported formats: MP4, AVI, MOV, MKV")
        print("\nUsage:")
        print("  text = await parse_video('video.mp4', extract_audio=True)")
        print("  metadata = await get_video_metadata('video.mp4')")
        print("\nFeatures:")
        print("  - Frame sampling and description")
        print("  - Audio track extraction and transcription (requires ffmpeg)")
        print("  - Video metadata (duration, FPS, resolution)")
        print("\nRequirements:")
        print("  - opencv-python (required)")
        print("  - ffmpeg binary (optional, for audio extraction)")
        print("  - openai-whisper (optional, for transcription)")
        print("\n✓ Demo complete!")

    asyncio.run(demo())
