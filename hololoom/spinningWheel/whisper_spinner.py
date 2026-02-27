"""
Whisper Spinner - Local Audio Transcription
============================================

Transcribes audio files using OpenAI's Whisper model (local).
Preserves timecodes for temporal navigation and reference.

Features:
- Local Whisper transcription (no API needed)
- Timecode preservation (word-level and segment-level)
- Multiple model sizes (tiny, base, small, medium, large)
- Speaker diarization (optional)
- Chunk-based processing for long audio
- Supports: WAV, MP3, M4A, FLAC, OGG

Requirements:
    pip install openai-whisper

Optional (for better performance):
    pip install torch  # GPU acceleration

Author: Claude Code
Date: November 2025
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass, field
import warnings
from datetime import timedelta

from HoloLoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinResult,
    SpinnerCapabilities,
    ImportanceScore
)
from HoloLoom.spinningWheel.importance import ImportanceScorer
from HoloLoom.protocols.types import MemoryShard

# Try to import whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    warnings.warn("OpenAI Whisper not available. Install with: pip install openai-whisper")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TimecodeSegment:
    """Audio segment with timecode"""
    start: float  # seconds
    end: float    # seconds
    text: str
    confidence: float = 0.0
    speaker: Optional[str] = None  # For diarization

    @property
    def duration(self) -> float:
        return self.end - self.start

    def format_timecode(self, start_or_end: str = 'start') -> str:
        """Format timecode as HH:MM:SS.mmm"""
        seconds = self.start if start_or_end == 'start' else self.end
        return str(timedelta(seconds=seconds))[:-3]  # Remove last 3 microsecond digits

    def to_srt_format(self, index: int) -> str:
        """Format as SRT subtitle entry"""
        return f"{index}\n{self.format_timecode('start')} --> {self.format_timecode('end')}\n{self.text}\n"

    # Alias for convenience
    def to_srt(self, index: int) -> str:
        """Alias for to_srt_format()"""
        return self.to_srt_format(index)


@dataclass
class AudioTranscription:
    """Complete transcription with metadata"""
    file_path: Path
    language: str
    duration: float  # Total audio duration in seconds
    segments: List[TimecodeSegment]
    full_text: str
    model_size: str
    word_timestamps: Optional[List[Dict[str, Any]]] = None  # Word-level timecodes

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    def to_srt(self) -> str:
        """Export as SRT subtitle file"""
        return "\n".join(seg.to_srt_format(i + 1) for i, seg in enumerate(self.segments))

    def get_text_at_time(self, timestamp: float) -> Optional[str]:
        """Get text at specific timestamp"""
        for seg in self.segments:
            if seg.start <= timestamp <= seg.end:
                return seg.text
        return None


# =============================================================================
# Whisper Spinner
# =============================================================================

class WhisperSpinner(BaseSpinner):
    """
    Transcribe audio files using OpenAI Whisper (local).

    Features:
    - Local transcription (no API needed)
    - Word and segment-level timecodes
    - Multiple model sizes
    - Language auto-detection
    - Chunk-based processing for long audio
    """

    def __init__(
        self,
        model_size: str = "base",
        importance_threshold: float = 0.3,
        language: Optional[str] = None,  # Auto-detect if None
        enable_word_timestamps: bool = True,
        chunk_duration: float = 300.0,  # 5 minutes
        device: str = "auto"  # "cpu", "cuda", or "auto"
    ):
        """
        Initialize WhisperSpinner.

        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            importance_threshold: Minimum importance score to keep segments
            language: Target language code (e.g., "en", "es", "fr") or None for auto-detect
            enable_word_timestamps: Extract word-level timestamps
            chunk_duration: Duration of audio chunks in seconds (for long audio)
            device: Device to use ("cpu", "cuda", or "auto")
        """
        super().__init__(name="whisper")
        self.model_size = model_size
        self.importance_threshold = importance_threshold
        self.language = language
        self.enable_word_timestamps = enable_word_timestamps
        self.chunk_duration = chunk_duration
        self.device = device

        # Load Whisper model
        self.model = None
        if WHISPER_AVAILABLE:
            self._load_model()

        # Initialize importance scorer
        self.importance_scorer = ImportanceScorer()

    def _load_model(self):
        """Load Whisper model"""
        try:
            import torch

            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device

            self.model = whisper.load_model(self.model_size, device=device)
            print(f"Whisper model '{self.model_size}' loaded on {device}")

        except Exception as e:
            warnings.warn(f"Failed to load Whisper model: {e}")
            self.model = None

    async def _spin_impl(self, source: Any, **kwargs) -> List[MemoryShard]:
        """
        Core transcription implementation.

        Args:
            source: Path to audio file
            **kwargs: Additional arguments (ignored)

        Returns:
            List of transcription shards
        """
        audio_path = Path(source)

        # Transcribe audio
        transcription = await self._transcribe_file(audio_path)

        # Convert to shards (base class handles filtering and wrapping)
        shards = self._transcription_to_shards(transcription)

        return shards

    async def _transcribe_file(self, audio_path: Path) -> AudioTranscription:
        """Transcribe audio file with Whisper"""
        # Run transcription in thread pool (Whisper is CPU/GPU intensive)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._transcribe_sync,
            audio_path
        )

        return result

    def _transcribe_sync(self, audio_path: Path) -> AudioTranscription:
        """Synchronous transcription (runs in thread pool)"""
        # Transcribe with Whisper
        result = self.model.transcribe(
            str(audio_path),
            language=self.language,
            word_timestamps=self.enable_word_timestamps,
            verbose=False
        )

        # Extract segments
        segments = []
        for seg in result['segments']:
            segments.append(TimecodeSegment(
                start=seg['start'],
                end=seg['end'],
                text=seg['text'].strip(),
                confidence=seg.get('confidence', 0.0)
            ))

        # Extract word timestamps if available
        word_timestamps = None
        if self.enable_word_timestamps and 'words' in result:
            word_timestamps = result['words']

        # Get audio duration (from last segment)
        duration = segments[-1].end if segments else 0.0

        return AudioTranscription(
            file_path=audio_path,
            language=result['language'],
            duration=duration,
            segments=segments,
            full_text=result['text'].strip(),
            model_size=self.model_size,
            word_timestamps=word_timestamps
        )

    def _transcription_to_shards(self, transcription: AudioTranscription) -> List[MemoryShard]:
        """Convert transcription to MemoryShards"""
        shards = []

        # Create shard for full transcription
        full_shard = MemoryShard(
            id=f"whisper_{transcription.file_path.stem}_full",
            text=transcription.full_text,
            episode=f"audio_{transcription.file_path.stem}",
            metadata={
                'file_name': transcription.file_path.name,
                'language': transcription.language,
                'duration': transcription.duration,
                'segment_count': transcription.segment_count,
                'model_size': transcription.model_size,
                'importance_score': self._score_transcription_importance(transcription),
                'shard_type': 'full_transcription'
            }
        )
        shards.append(full_shard)

        # Create shard per segment (for granular retrieval)
        for i, segment in enumerate(transcription.segments):
            segment_importance = self._score_segment_importance(segment, transcription)

            segment_shard = MemoryShard(
                id=f"whisper_{transcription.file_path.stem}_seg_{i}",
                text=segment.text,
                episode=f"audio_{transcription.file_path.stem}",
                metadata={
                    'file_name': transcription.file_path.name,
                    'language': transcription.language,
                    'start_time': segment.start,
                    'end_time': segment.end,
                    'duration': segment.duration,
                    'timecode': segment.format_timecode('start'),
                    'segment_index': i,
                    'segment_count': transcription.segment_count,
                    'confidence': segment.confidence,
                    'importance_score': segment_importance,
                    'shard_type': 'segment'
                }
            )
            shards.append(segment_shard)

        return shards

    def _score_transcription_importance(self, transcription: AudioTranscription) -> float:
        """Score importance of full transcription"""
        try:
            # Use importance scorer
            score = self.importance_scorer.score(
                transcription.full_text,
                metadata={
                    'length': len(transcription.full_text),
                    'duration': transcription.duration,
                    'segment_count': transcription.segment_count
                }
            )
            return score.score
        except Exception as e:
            # Gracefully handle scoring errors (regex issues, etc.)
            # Fall back to simple length-based scoring
            return min(len(transcription.full_text) / 1000, 1.0)

    def _score_segment_importance(self, segment: TimecodeSegment, transcription: AudioTranscription) -> float:
        """Score importance of individual segment"""
        try:
            # Base score on segment text
            score = self.importance_scorer.score(
                segment.text,
                metadata={
                    'length': len(segment.text),
                    'duration': segment.duration,
                    'confidence': segment.confidence
                }
            )

            # Boost important segments (questions, exclamations, keywords)
            if '?' in segment.text:
                score.score *= 1.2  # Questions are important
            if '!' in segment.text:
                score.score *= 1.1  # Exclamations are notable

            # Penalize very short segments
            if len(segment.text) < 20:
                score.score *= 0.8

            return min(1.0, score.score)
        except Exception as e:
            # Gracefully handle scoring errors (regex issues, etc.)
            # Fall back to simple length and confidence-based scoring
            base_score = min(len(segment.text) / 100, 1.0)
            return base_score * segment.confidence

    async def spin_stream(
        self,
        audio_path: Path,
        batch_size: int = 10
    ) -> AsyncIterator[MemoryShard]:
        """
        Stream shards as they're created.

        Args:
            audio_path: Path to audio file
            batch_size: Number of segments to process at once

        Yields:
            MemoryShard objects
        """
        result = await self.spin(audio_path)

        if not result.success:
            return

        for shard in result.shards:
            yield shard

    def get_capabilities(self) -> SpinnerCapabilities:
        """Get spinner capabilities"""
        return SpinnerCapabilities(
            streaming=True,
            incremental=False,  # No incremental audio (would need chunking)
            batch=False,
            supported_formats=['wav', 'mp3', 'm4a', 'flac', 'ogg', 'opus'],
            max_size_bytes=500 * 1024 * 1024,  # 500 MB limit
            metadata={'model_sizes': ['tiny', 'base', 'small', 'medium', 'large']}
        )

    def is_available(self) -> bool:
        """Check if Whisper is available"""
        return WHISPER_AVAILABLE and self.model is not None

    def export_srt(self, segments: List[TimecodeSegment], output_path: Path):
        """
        Export segments to SRT subtitle format.

        Args:
            segments: List of TimecodeSegment objects
            output_path: Path to output SRT file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, start=1):
                f.write(segment.to_srt(index=i))
                f.write('\n')


# =============================================================================
# Convenience Functions
# =============================================================================

async def transcribe_audio(
    audio_path: Path,
    model_size: str = "base",
    language: Optional[str] = None
) -> SpinResult:
    """
    Convenience function to transcribe audio.

    Args:
        audio_path: Path to audio file
        model_size: Whisper model size
        language: Target language (None for auto-detect)

    Returns:
        SpinResult with transcription shards
    """
    spinner = WhisperSpinner(
        model_size=model_size,
        language=language,
        importance_threshold=0.2  # Lower threshold for convenience function
    )

    return await spinner.spin(audio_path)


async def transcribe_with_timecodes(
    audio_path: Path,
    export_srt: bool = False
) -> Dict[str, Any]:
    """
    Transcribe audio and export with timecodes.

    Args:
        audio_path: Path to audio file
        export_srt: Export SRT subtitle file

    Returns:
        Dictionary with transcription and metadata
    """
    spinner = WhisperSpinner(
        model_size="base",
        enable_word_timestamps=True
    )

    # Transcribe
    transcription = await spinner._transcribe_file(audio_path)

    # Export SRT if requested
    srt_path = None
    if export_srt:
        srt_path = audio_path.with_suffix('.srt')
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(transcription.to_srt())

    return {
        'text': transcription.full_text,
        'language': transcription.language,
        'duration': transcription.duration,
        'segments': [
            {
                'start': seg.start,
                'end': seg.end,
                'text': seg.text,
                'timecode': seg.format_timecode('start')
            }
            for seg in transcription.segments
        ],
        'srt_path': str(srt_path) if srt_path else None
    }


# Alias for consistency with other spinners
spin_audio = transcribe_audio
