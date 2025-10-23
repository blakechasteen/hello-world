# -*- coding: utf-8 -*-
"""
YouTube Spinner
===============
Input adapter for ingesting YouTube video transcripts as MemoryShards.

Converts YouTube video captions/transcripts into structured memory shards
that can be processed by the HoloLoom orchestrator.

Design Philosophy:
- Thin adapter that normalizes YouTube data → MemoryShards
- Extracts video metadata (title, duration, language)
- Optionally segments long transcripts into chunks
- Optional enrichment for entity/motif extraction

Usage:
    from HoloLoom.spinningWheel.youtube import YouTubeSpinner, YouTubeSpinnerConfig

    config = YouTubeSpinnerConfig(
        chunk_duration=60.0,  # Segment into 60-second chunks
        enable_enrichment=True
    )

    spinner = YouTubeSpinner(config)
    shards = await spinner.spin({
        'url': 'https://www.youtube.com/watch?v=VIDEO_ID',
        'languages': ['en']
    })
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs

from .base import BaseSpinner, SpinnerConfig

# Import HoloLoom types
try:
    from HoloLoom.Documentation.types import MemoryShard
except ImportError:
    # Fallback if types not available
    from dataclasses import dataclass, field

    @dataclass
    class MemoryShard:
        id: str
        text: str
        episode: Optional[str] = None
        entities: List[str] = field(default_factory=list)
        motifs: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class YouTubeSpinnerConfig(SpinnerConfig):
    """
    Configuration for YouTube spinner.

    Attributes:
        chunk_duration: Split transcript into chunks of this duration (seconds).
                       If None, creates one shard per video.
        include_timestamps: Include timestamp info in shard metadata
        extract_entities: Extract basic entities from video text
    """
    chunk_duration: Optional[float] = None  # None = single shard per video
    include_timestamps: bool = True
    extract_entities: bool = True


class YouTubeTranscriptExtractor:
    """
    Low-level YouTube transcript extraction utility.
    Wraps youtube-transcript-api with error handling.
    """

    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """
        Extract video ID from various YouTube URL formats.

        Supports:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
        - https://www.youtube.com/shorts/VIDEO_ID
        - VIDEO_ID (direct ID)

        Args:
            url: YouTube URL or video ID

        Returns:
            Video ID string or None if invalid
        """
        # If it's already just an ID (11 characters, alphanumeric)
        if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
            return url

        # Parse various YouTube URL formats
        parsed = urlparse(url)

        if parsed.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
            if parsed.path == '/watch':
                return parse_qs(parsed.query).get('v', [None])[0]
            elif parsed.path.startswith('/embed/'):
                return parsed.path.split('/')[2]
            elif parsed.path.startswith('/shorts/'):
                return parsed.path.split('/')[2]
        elif parsed.hostname in ('youtu.be',):
            return parsed.path[1:]

        return None

    @staticmethod
    def get_transcript(video_id: str, languages: List[str] = None) -> Dict[str, Any]:
        """
        Fetch transcript for a YouTube video.

        Args:
            video_id: YouTube video ID
            languages: List of language codes to try (e.g., ['en', 'es'])
                      If None, tries English first, then any available

        Returns:
            Dict with 'text', 'language', 'is_generated', 'segments', 'duration'
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ImportError(
                "youtube-transcript-api is required. Install with:\n"
                "  pip install youtube-transcript-api"
            )

        if languages is None:
            languages = ['en']

        try:
            # Create API instance and fetch transcript
            api = YouTubeTranscriptApi()

            # List available transcripts
            transcript_list = api.list(video_id)

            # Try to find transcript in preferred language
            selected_transcript = None
            for transcript in transcript_list:
                if transcript.language_code in languages:
                    selected_transcript = transcript
                    break

            # If no preferred language found, use first available
            if selected_transcript is None and transcript_list:
                selected_transcript = transcript_list[0]

            if selected_transcript is None:
                raise Exception(f"No transcripts available for video: {video_id}")

            # Fetch the transcript segments
            segments = api.fetch(video_id, languages=[selected_transcript.language_code])

            # Convert segments to dicts if they're objects (newer API versions)
            segment_dicts = []
            for seg in segments:
                if isinstance(seg, dict):
                    segment_dicts.append(seg)
                else:
                    # Convert FetchedTranscriptSnippet object to dict
                    segment_dicts.append({
                        'text': seg.text,
                        'start': seg.start,
                        'duration': seg.duration
                    })

            # Combine all text
            full_text = ' '.join([seg['text'] for seg in segment_dicts])

            # Calculate total duration
            duration = 0
            if segment_dicts:
                last_segment = segment_dicts[-1]
                duration = last_segment['start'] + last_segment.get('duration', 0)

            return {
                'text': full_text,
                'language': selected_transcript.language_code,
                'is_generated': selected_transcript.is_generated,
                'segments': segment_dicts,
                'duration': duration,
                'video_id': video_id
            }

        except Exception as e:
            raise Exception(
                f"Could not retrieve transcript for video: {video_id}. "
                f"Error: {str(e)}\n"
                f"Note: This may fail in restricted environments (403 errors) "
                f"or if transcripts are disabled for the video."
            )


class YouTubeSpinner(BaseSpinner):
    """
    Spinner for YouTube video transcripts.

    Converts YouTube transcript data into MemoryShards suitable for
    HoloLoom processing.
    """

    def __init__(self, config: YouTubeSpinnerConfig = None):
        if config is None:
            config = YouTubeSpinnerConfig()
        super().__init__(config)
        self.config: YouTubeSpinnerConfig = config
        self.extractor = YouTubeTranscriptExtractor()

    async def spin(self, raw_data: Dict[str, Any]) -> List[MemoryShard]:
        """
        Convert YouTube video transcript → MemoryShards.

        Args:
            raw_data: Dict with keys:
                - 'url': YouTube URL or video ID (required)
                - 'languages': List of preferred language codes (optional)
                - 'episode': Episode/session identifier (optional)
                - 'metadata': Additional metadata to include (optional)

        Returns:
            List of MemoryShard objects
        """
        # Extract video ID
        url = raw_data.get('url')
        if not url:
            raise ValueError("'url' is required in raw_data")

        video_id = self.extractor.extract_video_id(url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {url}")

        # Get transcript
        languages = raw_data.get('languages', ['en'])
        transcript_data = self.extractor.get_transcript(video_id, languages)

        # Determine episode identifier
        episode = raw_data.get('episode', f"youtube_{video_id}")

        # Create shards
        if self.config.chunk_duration:
            # Split into time-based chunks
            shards = self._create_chunked_shards(transcript_data, episode, raw_data)
        else:
            # Single shard for entire video
            shards = self._create_single_shard(transcript_data, episode, raw_data)

        # Optional enrichment
        if self.config.enable_enrichment:
            shards = await self._enrich_shards(shards)

        return shards

    def _create_single_shard(
        self,
        transcript_data: Dict[str, Any],
        episode: str,
        raw_data: Dict[str, Any]
    ) -> List[MemoryShard]:
        """Create a single shard for the entire video transcript."""
        video_id = transcript_data['video_id']

        # Extract basic entities if configured
        entities = []
        if self.config.extract_entities:
            entities = self._extract_basic_entities(transcript_data['text'])

        # Build metadata
        metadata = {
            'source': 'youtube',
            'video_id': video_id,
            'language': transcript_data['language'],
            'is_generated': transcript_data['is_generated'],
            'duration': transcript_data['duration'],
            'segment_count': len(transcript_data['segments']),
            'url': f"https://www.youtube.com/watch?v={video_id}"
        }

        # Include timestamps if configured
        if self.config.include_timestamps and transcript_data['segments']:
            metadata['timestamps'] = [
                {
                    'start': seg['start'],
                    'duration': seg.get('duration', 0),
                    'text': seg['text']
                }
                for seg in transcript_data['segments']
            ]

        # Merge additional metadata from raw_data
        if 'metadata' in raw_data:
            metadata.update(raw_data['metadata'])

        shard = MemoryShard(
            id=f"{episode}_full",
            text=transcript_data['text'],
            episode=episode,
            entities=entities,
            motifs=[],  # Will be populated by orchestrator's motif detector
            metadata=metadata
        )

        return [shard]

    def _create_chunked_shards(
        self,
        transcript_data: Dict[str, Any],
        episode: str,
        raw_data: Dict[str, Any]
    ) -> List[MemoryShard]:
        """Split transcript into time-based chunks."""
        video_id = transcript_data['video_id']
        segments = transcript_data['segments']
        chunk_duration = self.config.chunk_duration

        shards = []
        current_chunk_text = []
        current_chunk_segments = []
        chunk_start_time = 0
        chunk_index = 0

        for segment in segments:
            seg_start = segment['start']

            # Check if we should start a new chunk
            if seg_start >= chunk_start_time + chunk_duration and current_chunk_text:
                # Finalize current chunk
                chunk_text = ' '.join(current_chunk_text)
                entities = []
                if self.config.extract_entities:
                    entities = self._extract_basic_entities(chunk_text)

                metadata = {
                    'source': 'youtube',
                    'video_id': video_id,
                    'language': transcript_data['language'],
                    'is_generated': transcript_data['is_generated'],
                    'chunk_index': chunk_index,
                    'chunk_start': chunk_start_time,
                    'chunk_end': seg_start,
                    'url': f"https://www.youtube.com/watch?v={video_id}&t={int(chunk_start_time)}"
                }

                if self.config.include_timestamps:
                    metadata['timestamps'] = current_chunk_segments

                # Merge additional metadata
                if 'metadata' in raw_data:
                    metadata.update(raw_data['metadata'])

                shard = MemoryShard(
                    id=f"{episode}_chunk_{chunk_index:03d}",
                    text=chunk_text,
                    episode=episode,
                    entities=entities,
                    motifs=[],
                    metadata=metadata
                )
                shards.append(shard)

                # Start new chunk
                current_chunk_text = []
                current_chunk_segments = []
                chunk_start_time = seg_start
                chunk_index += 1

            # Add segment to current chunk
            current_chunk_text.append(segment['text'])
            if self.config.include_timestamps:
                current_chunk_segments.append({
                    'start': segment['start'],
                    'duration': segment.get('duration', 0),
                    'text': segment['text']
                })

        # Finalize last chunk
        if current_chunk_text:
            chunk_text = ' '.join(current_chunk_text)
            entities = []
            if self.config.extract_entities:
                entities = self._extract_basic_entities(chunk_text)

            metadata = {
                'source': 'youtube',
                'video_id': video_id,
                'language': transcript_data['language'],
                'is_generated': transcript_data['is_generated'],
                'chunk_index': chunk_index,
                'chunk_start': chunk_start_time,
                'chunk_end': transcript_data['duration'],
                'url': f"https://www.youtube.com/watch?v={video_id}&t={int(chunk_start_time)}"
            }

            if self.config.include_timestamps:
                metadata['timestamps'] = current_chunk_segments

            if 'metadata' in raw_data:
                metadata.update(raw_data['metadata'])

            shard = MemoryShard(
                id=f"{episode}_chunk_{chunk_index:03d}",
                text=chunk_text,
                episode=episode,
                entities=entities,
                motifs=[],
                metadata=metadata
            )
            shards.append(shard)

        return shards

    def _extract_basic_entities(self, text: str) -> List[str]:
        """
        Extract basic entities using simple heuristics.

        More sophisticated extraction happens during enrichment or
        in the orchestrator's motif detection phase.
        """
        entities = []

        # Extract capitalized words (potential proper nouns)
        # Skip common words
        common_words = {'The', 'This', 'That', 'These', 'Those', 'I', 'You', 'We', 'They'}
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

        for word in words:
            if word not in common_words and len(word) > 2:
                entities.append(word)

        # Deduplicate and limit
        entities = list(set(entities))[:20]  # Keep top 20 unique entities

        return entities

    async def _enrich_shards(self, shards: List[MemoryShard]) -> List[MemoryShard]:
        """
        Enrich shards using optional enrichment services.
        Delegates to parent class enrichment infrastructure.
        """
        for shard in shards:
            enrichment = await self.enrich(shard.text)

            # Merge enrichment results into shard
            if enrichment:
                # Update entities from enrichment
                if 'ollama' in enrichment and 'entities' in enrichment['ollama']:
                    shard.entities.extend(enrichment['ollama']['entities'])
                    shard.entities = list(set(shard.entities))[:30]  # Dedupe and limit

                # Update motifs from enrichment
                if 'ollama' in enrichment and 'motifs' in enrichment['ollama']:
                    shard.motifs.extend(enrichment['ollama']['motifs'])
                    shard.motifs = list(set(shard.motifs))

                # Store enrichment in metadata
                if shard.metadata is None:
                    shard.metadata = {}
                shard.metadata['enrichment'] = enrichment

        return shards


# Convenience functions for quick usage
async def transcribe_youtube(
    url: str,
    languages: List[str] = None,
    chunk_duration: Optional[float] = None,
    enable_enrichment: bool = False
) -> List[MemoryShard]:
    """
    Quick function to transcribe a YouTube video into MemoryShards.

    Args:
        url: YouTube URL or video ID
        languages: Preferred language codes (default: ['en'])
        chunk_duration: Split into chunks of this duration (default: None = single shard)
        enable_enrichment: Enable Ollama enrichment (default: False)

    Returns:
        List of MemoryShard objects
    """
    config = YouTubeSpinnerConfig(
        chunk_duration=chunk_duration,
        enable_enrichment=enable_enrichment
    )

    spinner = YouTubeSpinner(config)

    raw_data = {
        'url': url,
        'languages': languages or ['en']
    }

    return await spinner.spin(raw_data)
