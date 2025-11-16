"""
YouTube Spinner - Video Transcription Ingestion
================================================

Ingests YouTube video transcripts with timecodes and metadata.
Supports multiple URL formats, language preferences, and chunking.

Features:
- Multiple URL format support (youtube.com, youtu.be, shorts, embed)
- Automatic transcript retrieval
- Language preference with fallback
- Time-based chunking for long videos
- Timecode preservation
- Video metadata extraction
- Zero API key required (uses youtube-transcript-api)

Requirements:
    pip install youtube-transcript-api

Optional (for metadata):
    pip install pytube

Author: Claude Code
Date: November 2025
"""

import re
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass, field
from urllib.parse import urlparse, parse_qs
import warnings

from HoloLoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinResult,
    SpinnerCapabilities
)
from HoloLoom.spinningWheel.importance import ImportanceScorer
from HoloLoom.Documentation.types import MemoryShard

# Try to import youtube-transcript-api
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable
    )
    TRANSCRIPT_API_AVAILABLE = True
except ImportError:
    TRANSCRIPT_API_AVAILABLE = False
    warnings.warn("youtube-transcript-api not available. Install with: pip install youtube-transcript-api")

# Try to import pytube for metadata
try:
    from pytube import YouTube
    PYTUBE_AVAILABLE = True
except ImportError:
    PYTUBE_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TranscriptSegment:
    """Transcript segment with timecode"""
    text: str
    start: float  # seconds
    duration: float  # seconds

    @property
    def end(self) -> float:
        return self.start + self.duration

    def format_timecode(self) -> str:
        """Format start time as MM:SS"""
        minutes = int(self.start // 60)
        seconds = int(self.start % 60)
        return f"{minutes:02d}:{seconds:02d}"


@dataclass
class VideoMetadata:
    """YouTube video metadata"""
    video_id: str
    title: Optional[str] = None
    author: Optional[str] = None
    description: Optional[str] = None
    length: Optional[int] = None  # seconds
    views: Optional[int] = None
    publish_date: Optional[str] = None


@dataclass
class YouTubeTranscription:
    """Complete YouTube transcription"""
    video_id: str
    metadata: VideoMetadata
    segments: List[TranscriptSegment]
    language: str
    full_text: str

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def duration(self) -> float:
        if not self.segments:
            return 0.0
        return self.segments[-1].end


# =============================================================================
# YouTube Spinner
# =============================================================================

class YouTubeSpinner(BaseSpinner):
    """
    Ingest YouTube video transcripts.

    Features:
    - Multi-format URL support
    - Language preference with fallback
    - Time-based chunking
    - Metadata extraction
    """

    def __init__(
        self,
        importance_threshold: float = 0.3,
        languages: List[str] = ['en'],  # Preference order
        chunk_duration: Optional[float] = None,  # Seconds per chunk (None = full video)
        enable_metadata: bool = True
    ):
        """
        Initialize YouTubeSpinner.

        Args:
            importance_threshold: Minimum importance score
            languages: Language preferences in order (e.g., ['en', 'es', 'auto'])
            chunk_duration: Duration of chunks in seconds (None for full video)
            enable_metadata: Extract video metadata (requires pytube)
        """
        super().__init__(name="youtube")
        self.importance_threshold = importance_threshold
        self.languages = languages
        self.chunk_duration = chunk_duration
        self.enable_metadata = enable_metadata

        # Initialize importance scorer
        self.importance_scorer = ImportanceScorer()

    async def _spin_impl(self, source: Any, **kwargs) -> List[MemoryShard]:
        """
        Core YouTube transcription implementation.

        Args:
            source: YouTube URL or video ID
            **kwargs: Additional arguments (ignored)

        Returns:
            List of transcript shards

        Raises:
            ValueError: Invalid URL or video ID
            TranscriptsDisabled: Transcripts disabled for this video
            NoTranscriptFound: No transcript available
            VideoUnavailable: Video is unavailable
        """
        url_or_id = str(source)

        # Extract video ID
        video_id = self._extract_video_id(url_or_id)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL or video ID: {url_or_id}")

        # Get transcription
        transcription = await self._get_transcription(video_id)

        # Convert to shards (base class handles filtering and wrapping)
        shards = self._transcription_to_shards(transcription)

        return shards

    def _extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract video ID from various YouTube URL formats.

        Supports:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
        - https://www.youtube.com/shorts/VIDEO_ID
        - VIDEO_ID (direct ID)
        """
        # If it's already just an ID (11 characters, alphanumeric)
        if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
            return url

        # Parse URL
        parsed = urlparse(url)

        # Format 1: youtube.com/watch?v=VIDEO_ID
        if 'youtube.com' in parsed.netloc and '/watch' in parsed.path:
            query_params = parse_qs(parsed.query)
            if 'v' in query_params:
                return query_params['v'][0]

        # Format 2: youtu.be/VIDEO_ID
        if 'youtu.be' in parsed.netloc:
            video_id = parsed.path.lstrip('/')
            return video_id if video_id else None

        # Format 3: youtube.com/embed/VIDEO_ID
        if 'youtube.com' in parsed.netloc and '/embed/' in parsed.path:
            parts = parsed.path.split('/embed/')
            if len(parts) > 1:
                return parts[1].split('?')[0]  # Remove query params

        # Format 4: youtube.com/shorts/VIDEO_ID
        if 'youtube.com' in parsed.netloc and '/shorts/' in parsed.path:
            parts = parsed.path.split('/shorts/')
            if len(parts) > 1:
                return parts[1].split('?')[0]

        return None

    async def _get_transcription(self, video_id: str) -> YouTubeTranscription:
        """Get transcript and metadata for video"""
        # Run in thread pool (API calls are blocking)
        loop = asyncio.get_event_loop()

        # Get transcript
        transcript = await loop.run_in_executor(
            None,
            self._fetch_transcript,
            video_id
        )

        # Get metadata if enabled
        metadata = VideoMetadata(video_id=video_id)
        if self.enable_metadata and PYTUBE_AVAILABLE:
            metadata = await loop.run_in_executor(
                None,
                self._fetch_metadata,
                video_id
            )

        # Parse transcript
        segments = [
            TranscriptSegment(
                text=entry['text'].strip(),
                start=entry['start'],
                duration=entry['duration']
            )
            for entry in transcript
        ]

        # Get full text
        full_text = ' '.join(seg.text for seg in segments)

        # Detect language
        language = transcript[0].get('language', 'unknown') if transcript else 'unknown'

        return YouTubeTranscription(
            video_id=video_id,
            metadata=metadata,
            segments=segments,
            language=language,
            full_text=full_text
        )

    def _fetch_transcript(self, video_id: str) -> List[Dict[str, Any]]:
        """Fetch transcript (sync function for thread pool)"""
        # Try each language in order
        for lang in self.languages:
            try:
                if lang == 'auto':
                    # Get auto-generated transcript
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                    transcript = transcript_list.find_generated_transcript(self.languages)
                else:
                    # Get transcript in specific language
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                    return transcript
            except (NoTranscriptFound, Exception):
                continue

        # Fallback: try any available transcript
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_list.find_transcript(self.languages)
            return transcript.fetch()
        except Exception as e:
            raise NoTranscriptFound(f"No transcript found in languages: {self.languages}")

    def _fetch_metadata(self, video_id: str) -> VideoMetadata:
        """Fetch video metadata (sync function for thread pool)"""
        try:
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")

            return VideoMetadata(
                video_id=video_id,
                title=yt.title,
                author=yt.author,
                description=yt.description,
                length=yt.length,
                views=yt.views,
                publish_date=yt.publish_date.isoformat() if yt.publish_date else None
            )
        except Exception as e:
            warnings.warn(f"Failed to fetch metadata: {e}")
            return VideoMetadata(video_id=video_id)

    def _transcription_to_shards(self, transcription: YouTubeTranscription) -> List[MemoryShard]:
        """Convert transcription to MemoryShards"""
        shards = []

        if self.chunk_duration is None:
            # Single shard for entire video
            shard = MemoryShard(
                id=f"youtube_{transcription.video_id}",
                text=transcription.full_text,
                episode=f"youtube_{transcription.video_id}",
                metadata={
                    'video_id': transcription.video_id,
                    'title': transcription.metadata.title,
                    'author': transcription.metadata.author,
                    'language': transcription.language,
                    'duration': transcription.duration,
                    'segment_count': transcription.segment_count,
                    'url': f"https://www.youtube.com/watch?v={transcription.video_id}",
                    'importance_score': self._score_transcription_importance(transcription),
                    'shard_type': 'full_video'
                }
            )
            shards.append(shard)

        else:
            # Create chunks based on duration
            current_chunk = []
            chunk_start_time = 0.0
            chunk_index = 0

            for segment in transcription.segments:
                current_chunk.append(segment)

                # Check if chunk duration reached
                if segment.end - chunk_start_time >= self.chunk_duration:
                    # Create shard for chunk
                    chunk_text = ' '.join(seg.text for seg in current_chunk)
                    chunk_shard = MemoryShard(
                        id=f"youtube_{transcription.video_id}_chunk_{chunk_index}",
                        text=chunk_text,
                        episode=f"youtube_{transcription.video_id}",
                        metadata={
                            'video_id': transcription.video_id,
                            'title': transcription.metadata.title,
                            'language': transcription.language,
                            'chunk_index': chunk_index,
                            'start_time': chunk_start_time,
                            'end_time': segment.end,
                            'duration': segment.end - chunk_start_time,
                            'timecode': f"{int(chunk_start_time // 60):02d}:{int(chunk_start_time % 60):02d}",
                            'url': f"https://www.youtube.com/watch?v={transcription.video_id}&t={int(chunk_start_time)}s",
                            'importance_score': self._score_chunk_importance(chunk_text),
                            'shard_type': 'chunk'
                        }
                    )
                    shards.append(chunk_shard)

                    # Reset for next chunk
                    current_chunk = []
                    chunk_start_time = segment.end
                    chunk_index += 1

            # Add remaining segments as final chunk
            if current_chunk:
                chunk_text = ' '.join(seg.text for seg in current_chunk)
                chunk_shard = MemoryShard(
                    id=f"youtube_{transcription.video_id}_chunk_{chunk_index}",
                    text=chunk_text,
                    episode=f"youtube_{transcription.video_id}",
                    metadata={
                        'video_id': transcription.video_id,
                        'title': transcription.metadata.title,
                        'language': transcription.language,
                        'chunk_index': chunk_index,
                        'start_time': chunk_start_time,
                        'end_time': transcription.duration,
                        'timecode': f"{int(chunk_start_time // 60):02d}:{int(chunk_start_time % 60):02d}",
                        'url': f"https://www.youtube.com/watch?v={transcription.video_id}&t={int(chunk_start_time)}s",
                        'importance_score': self._score_chunk_importance(chunk_text),
                        'shard_type': 'chunk'
                    }
                )
                shards.append(chunk_shard)

        return shards

    def _score_transcription_importance(self, transcription: YouTubeTranscription) -> float:
        """Score importance of full transcription"""
        score = self.importance_scorer.score(
            transcription.full_text,
            metadata={
                'length': len(transcription.full_text),
                'duration': transcription.duration
            }
        )
        return score.score

    def _score_chunk_importance(self, chunk_text: str) -> float:
        """Score importance of chunk"""
        score = self.importance_scorer.score(
            chunk_text,
            metadata={'length': len(chunk_text)}
        )
        return score.score

    async def spin_stream(
        self,
        url_or_id: str,
        batch_size: int = 1
    ) -> AsyncIterator[MemoryShard]:
        """
        Stream shards as they're created.

        Args:
            url_or_id: YouTube URL or video ID
            batch_size: Not used (kept for API compatibility)

        Yields:
            MemoryShard objects
        """
        result = await self.spin(url_or_id)

        if not result.success:
            return

        for shard in result.shards:
            yield shard

    def get_capabilities(self) -> SpinnerCapabilities:
        """Get spinner capabilities"""
        return SpinnerCapabilities(
            basic_processing=True,
            streaming=False,  # No true streaming support
            incremental=False,
            batch_processing=True,  # Can process multiple URLs
            importance_scoring=True,  # Has importance scorer
            entity_extraction=False,
            motif_extraction=True,
            supported_formats=['youtube_url', 'youtube_id']
        )

    def is_available(self) -> bool:
        """Check if YouTube transcript API is available"""
        return TRANSCRIPT_API_AVAILABLE


# =============================================================================
# Convenience Functions
# =============================================================================

async def transcribe_youtube(
    url_or_id: str,
    chunk_duration: Optional[float] = None,
    languages: List[str] = ['en']
) -> SpinResult:
    """
    Convenience function to transcribe YouTube video.

    Args:
        url_or_id: YouTube URL or video ID
        chunk_duration: Duration of chunks in seconds (None for full video)
        languages: Language preferences

    Returns:
        SpinResult with transcript shards
    """
    spinner = YouTubeSpinner(
        importance_threshold=0.2,  # Lower threshold for convenience
        languages=languages,
        chunk_duration=chunk_duration
    )

    return await spinner.spin(url_or_id)


# Alias for consistency with other spinners
spin_youtube = transcribe_youtube
