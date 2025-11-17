"""
Audio Parser - Extract Content from Audio Files
================================================
Parse audio files and extract transcriptions and metadata.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try imports with graceful fallback
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    logger.warning("whisper not installed. Audio transcription unavailable.")
    WHISPER_AVAILABLE = False
    whisper = None

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    logger.warning("librosa not installed. Audio metadata unavailable.")
    LIBROSA_AVAILABLE = False
    librosa = None
    sf = None


async def parse_audio(
    file_path: str,
    model_name: str = "base",
    language: Optional[str] = None
) -> str:
    """
    Extract transcription and metadata from audio file.

    Args:
        file_path: Path to audio file
        model_name: Whisper model size ("tiny", "base", "small", "medium", "large")
        language: Language code (None for auto-detect)

    Returns:
        Transcribed text with metadata

    Raises:
        RuntimeError: If Whisper/librosa not available
        FileNotFoundError: If audio file not found
    """
    if not WHISPER_AVAILABLE:
        raise RuntimeError("Audio parsing requires whisper: pip install openai-whisper")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {path}")

    text_parts = []

    try:
        # Load Whisper model
        logger.info(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)

        # Load audio metadata
        duration = None
        sample_rate = None
        if LIBROSA_AVAILABLE:
            try:
                # Get duration without loading full audio
                duration = librosa.get_duration(path=str(path))
                # Load for sample rate
                y, sr = librosa.load(str(path), sr=None, duration=1.0)
                sample_rate = sr
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")

        # Add header
        text_parts.append(f"[Audio: {path.name}]")
        if duration:
            text_parts.append(f"Duration: {duration:.2f} seconds")
        if sample_rate:
            text_parts.append(f"Sample Rate: {sample_rate} Hz")

        # Transcribe
        logger.info(f"Transcribing {path.name}...")
        result = model.transcribe(
            str(path),
            language=language,
            fp16=False  # Use FP32 for CPU compatibility
        )

        # Extract text
        transcription = result.get("text", "").strip()
        detected_lang = result.get("language", "unknown")

        text_parts.append(f"Language: {detected_lang}")
        text_parts.append("\nTranscription:")
        text_parts.append(transcription)

        # Add segments if available
        segments = result.get("segments", [])
        if segments and len(segments) > 1:
            text_parts.append(f"\n[{len(segments)} segments]")

        # Combine all parts
        full_text = "\n".join(text_parts)

        logger.info(
            f"Transcribed {path.name}: {len(transcription)} chars, "
            f"language={detected_lang}"
        )

        return full_text

    except Exception as e:
        logger.error(f"Failed to parse audio {file_path}: {e}")
        raise


async def get_audio_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from audio file.

    Args:
        file_path: Path to audio file

    Returns:
        Dict with metadata (duration, sample_rate, channels, etc.)
    """
    if not LIBROSA_AVAILABLE:
        return {}

    path = Path(file_path)
    if not path.exists():
        return {}

    try:
        # Get duration
        duration = librosa.get_duration(path=str(path))

        # Load audio
        y, sr = librosa.load(str(path), sr=None, mono=False)

        # Determine channels
        if y.ndim == 1:
            channels = 1
        else:
            channels = y.shape[0]

        metadata = {
            "duration": duration,
            "sample_rate": sr,
            "channels": channels,
            "samples": len(y) if channels == 1 else y.shape[1],
            "format": path.suffix[1:].upper()  # Extension as format
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
        print("Audio Parser Demo")
        print("=" * 80 + "\n")

        if not WHISPER_AVAILABLE:
            print("Whisper not available")
            print("Install with: pip install openai-whisper")
            return

        print("Audio parser is ready")
        print("Supported formats: MP3, WAV, FLAC, M4A, OGG")
        print("\nUsage:")
        print("  text = await parse_audio('speech.mp3', model_name='base')")
        print("  metadata = await get_audio_metadata('speech.mp3')")
        print("\nWhisper models: tiny, base, small, medium, large")
        print("  - tiny: fastest, lowest quality")
        print("  - base: good balance (default)")
        print("  - large: best quality, slowest")
        print("\n✓ Demo complete!")

    asyncio.run(demo())
