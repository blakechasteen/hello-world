"""
Playlist Spinner — YouTube Playlist Ingestion with Resonance Detection
======================================================================

Scrapes YouTube playlists for new videos, fetches transcripts via the
existing YouTubeSpinner, and tracks cross-playlist resonance (videos
appearing on multiple playlists score higher).

The resonance signal is source-agnostic: it counts how many distinct
sources contain the same canonical URL. YouTube playlists are the first
source type, but Pocket/Raindrop/RSS can plug into the same scorer.

Requirements:
    pip install yt-dlp youtube-transcript-api

Optional (for metadata):
    pip install pytube

Author: Claude Code
Date: March 2026
"""

import asyncio
import json
import logging
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hololoom.protocols.types import MemoryShard
from hololoom.spinningWheel.importance import ImportanceScorer
from hololoom.spinningWheel.protocol import (
    BaseSpinner,
    ImportanceScore,
    ImportanceSignals,
    SpinnerCapabilities,
    SpinnerCheckpoint,
    SpinResult,
)

logger = logging.getLogger(__name__)

# yt-dlp for playlist manifest fetching (no API key needed)
try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False
    warnings.warn("yt-dlp not available. Install with: pip install yt-dlp")

# Reuse existing YouTubeSpinner for transcript fetching
try:
    from hololoom.spinningWheel.youtube_spinner import YouTubeSpinner
    YOUTUBE_SPINNER_AVAILABLE = True
except ImportError:
    YOUTUBE_SPINNER_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PlaylistEntry:
    """A single video entry from a playlist manifest."""
    video_id: str
    title: str | None = None
    channel: str | None = None
    duration: float | None = None  # seconds
    upload_date: str | None = None  # YYYYMMDD from yt-dlp
    playlist_index: int = 0


@dataclass
class PlaylistManifest:
    """Metadata + entries for one playlist."""
    playlist_id: str
    title: str | None = None
    entries: list[PlaylistEntry] = field(default_factory=list)
    fetched_at: float = field(default_factory=time.time)


@dataclass
class ResonanceIndex:
    """
    Tracks which sources contain each video.

    The resonance count (number of distinct sources containing a URL)
    is the core importance signal. Source-agnostic by design — YouTube
    playlists are just the first source type.
    """
    # video_id → set of source_ids (e.g., playlist URLs)
    _sources: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    # video_id → metadata from first encounter
    _metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, video_id: str, source_id: str, metadata: dict[str, Any] | None = None):
        """Register a video as appearing in a source."""
        self._sources[video_id].add(source_id)
        if video_id not in self._metadata and metadata:
            self._metadata[video_id] = metadata

    def resonance_count(self, video_id: str) -> int:
        """How many distinct sources contain this video."""
        return len(self._sources.get(video_id, set()))

    def resonance_score(self, video_id: str, saturation: int = 3) -> float:
        """
        Normalized resonance score.

        Args:
            video_id: The video to score.
            saturation: Number of sources for max score (default 3).

        Returns:
            0.0–1.0, where saturation+ sources = 1.0.
        """
        count = self.resonance_count(video_id)
        return min(count / saturation, 1.0)

    def get_metadata(self, video_id: str) -> dict[str, Any]:
        return self._metadata.get(video_id, {})

    def all_video_ids(self) -> set[str]:
        return set(self._sources.keys())

    def ranked(self) -> list[tuple[str, int]]:
        """All videos ranked by resonance count, descending."""
        return sorted(
            ((vid, len(sources)) for vid, sources in self._sources.items()),
            key=lambda x: x[1],
            reverse=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'sources': {vid: list(srcs) for vid, srcs in self._sources.items()},
            'metadata': self._metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ResonanceIndex':
        idx = cls()
        for vid, srcs in data.get('sources', {}).items():
            idx._sources[vid] = set(srcs)
        idx._metadata = data.get('metadata', {})
        return idx


# =============================================================================
# Resonance Importance Scorer
# =============================================================================

class ResonanceScorer:
    """
    Source-agnostic importance scorer that weights cross-source resonance.

    Wraps the standard ImportanceScorer and injects a resonance signal
    as a custom scorer. The resonance signal dominates because curation
    behavior (adding to multiple playlists) is a stronger signal than
    any content heuristic.
    """

    # Weight distribution (sums to ~1.0 with custom signals)
    WEIGHTS = {
        'length': 0.10,
        'technical': 0.10,
        'structural': 0.05,
        'authority': 0.10,
        'recency': 0.15,
        'engagement': 0.10,
        'reference': 0.05,
    }

    def __init__(
        self,
        resonance_index: ResonanceIndex,
        resonance_weight: float = 0.30,
        density_weight: float = 0.05,
    ):
        self.resonance_index = resonance_index
        self.resonance_weight = resonance_weight
        self.density_weight = density_weight

        self._base_scorer = ImportanceScorer(
            custom_scorers={
                'resonance': self._resonance_signal,
                'transcript_density': self._density_signal,
            }
        )

    def _resonance_signal(self, text: str, metadata: dict | None) -> float:
        """Custom signal: cross-source resonance."""
        if not metadata:
            return 0.0
        video_id = metadata.get('video_id', '')
        return self.resonance_index.resonance_score(video_id)

    def _density_signal(self, text: str, metadata: dict | None) -> float:
        """Custom signal: transcript density (long substantive content)."""
        word_count = len(text.split())
        # 500+ words = full score, <50 = near zero
        return min(word_count / 500, 1.0)

    def score(
        self,
        text: str,
        video_id: str,
        channel: str | None = None,
        timestamp: float | None = None,
        engagement: dict[str, int] | None = None,
    ) -> ImportanceScore:
        """Score a video transcript with resonance-aware importance."""
        source = f"https://www.youtube.com/watch?v={video_id}"
        return self._base_scorer.score(
            text=text,
            source=source,
            timestamp=timestamp,
            author=channel,
            engagement=engagement,
            metadata={'video_id': video_id},
        )


# =============================================================================
# Playlist Spinner
# =============================================================================

class PlaylistSpinner(BaseSpinner):
    """
    Ingest YouTube playlists with cross-playlist resonance detection.

    Fetches playlist manifests via yt-dlp (no API key), deduplicates
    across playlists using video_id checkpoints, delegates transcript
    fetching to the existing YouTubeSpinner, and scores importance
    with the resonance signal (multi-playlist overlap) dominating.

    Features:
    - Multi-playlist batch ingestion
    - Cross-playlist resonance detection (videos on N playlists score higher)
    - Incremental: only processes new videos since last checkpoint
    - Backpressure: configurable batch size, high-resonance videos first
    - Graceful fallback when transcripts unavailable (title + description shard)
    """

    def __init__(
        self,
        importance_threshold: float = 0.2,
        batch_size: int = 20,
        transcript_languages: list[str] | None = None,
        checkpoint_dir: Path | None = None,
        resonance_saturation: int = 3,
    ):
        """
        Args:
            importance_threshold: Min importance to keep a shard.
            batch_size: Max videos to process per run (backpressure).
            transcript_languages: Language preference for transcripts.
            checkpoint_dir: Directory for checkpoint persistence.
            resonance_saturation: Sources needed for max resonance score.
        """
        super().__init__(
            name='playlist',
            importance_threshold=importance_threshold,
            checkpoint_dir=checkpoint_dir,
        )
        self.batch_size = batch_size
        self.transcript_languages = transcript_languages or ['en']
        self.resonance_saturation = resonance_saturation

        # Resonance index tracks cross-playlist overlap
        self._resonance_index = ResonanceIndex()

        # Lazy-init the YouTubeSpinner for transcript fetching
        self._yt_spinner: YouTubeSpinner | None = None

    def _get_yt_spinner(self) -> YouTubeSpinner | None:
        if self._yt_spinner is None and YOUTUBE_SPINNER_AVAILABLE:
            self._yt_spinner = YouTubeSpinner(
                importance_threshold=0.0,  # We do our own filtering
                languages=self.transcript_languages,
                chunk_duration=None,  # Full video shards
                enable_metadata=True,
            )
        return self._yt_spinner

    # =========================================================================
    # Protocol implementation
    # =========================================================================

    def get_capabilities(self) -> SpinnerCapabilities:
        return SpinnerCapabilities(
            basic_processing=True,
            streaming=False,
            incremental=True,
            batch_processing=True,
            importance_scoring=True,
            entity_extraction=False,
            motif_extraction=False,
            supported_formats=['youtube_playlist_url', 'youtube_playlist_id'],
        )

    def is_available(self) -> bool:
        return YTDLP_AVAILABLE

    # =========================================================================
    # Core spin
    # =========================================================================

    async def _spin_impl(self, source: Any, **kwargs) -> list[MemoryShard]:
        """
        Spin one or more playlist URLs.

        Args:
            source: A single playlist URL/ID, or a list of them.

        Returns:
            List of MemoryShards for new videos.
        """
        # Normalize input
        if isinstance(source, str):
            playlist_urls = [source]
        elif isinstance(source, (list, tuple)):
            playlist_urls = list(source)
        else:
            raise ValueError(f"Expected playlist URL(s), got {type(source)}")

        # Load checkpoint
        checkpoint = self._load_resonance_checkpoint()
        seen_ids: set[str] = set(checkpoint.state.get('seen_video_ids', []))

        # Phase 1: Fetch all playlist manifests
        manifests = await self._fetch_manifests(playlist_urls)

        # Phase 2: Build resonance index (all videos, even already-seen)
        for manifest in manifests:
            for entry in manifest.entries:
                self._resonance_index.add(
                    video_id=entry.video_id,
                    source_id=manifest.playlist_id,
                    metadata={
                        'title': entry.title,
                        'channel': entry.channel,
                        'duration': entry.duration,
                        'upload_date': entry.upload_date,
                    },
                )

        # Phase 3: Identify new videos (not in checkpoint)
        all_video_ids = self._resonance_index.all_video_ids()
        new_ids = all_video_ids - seen_ids

        if not new_ids:
            logger.info("No new videos found across %d playlists", len(manifests))
            return []

        # Phase 4: Rank new videos by resonance (high resonance first)
        ranked = sorted(
            new_ids,
            key=lambda vid: self._resonance_index.resonance_count(vid),
            reverse=True,
        )

        # Backpressure: cap at batch_size
        batch = ranked[:self.batch_size]
        if len(ranked) > self.batch_size:
            logger.info(
                "Backpressure: processing %d of %d new videos (high-resonance first)",
                self.batch_size, len(ranked),
            )

        # Phase 5: Fetch transcripts and build shards
        scorer = ResonanceScorer(
            resonance_index=self._resonance_index,
        )
        shards = await self._process_videos(batch, scorer)

        # Phase 6: Update checkpoint with processed video IDs
        seen_ids.update(batch)
        self._save_resonance_checkpoint(seen_ids, manifests)

        logger.info(
            "Processed %d videos → %d shards (resonance index: %d total videos)",
            len(batch), len(shards), len(all_video_ids),
        )

        return shards

    # =========================================================================
    # Incremental support
    # =========================================================================

    async def spin_incremental(
        self,
        source: Any,
        checkpoint: SpinnerCheckpoint | None = None,
        **kwargs,
    ) -> SpinResult:
        """Incremental spin — just delegates to spin() which is already incremental."""
        return await self.spin(source, **kwargs)

    # =========================================================================
    # Playlist manifest fetching
    # =========================================================================

    async def _fetch_manifests(
        self, playlist_urls: list[str]
    ) -> list[PlaylistManifest]:
        """Fetch playlist manifests concurrently via yt-dlp."""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self._fetch_one_manifest, url)
            for url in playlist_urls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        manifests = []
        for url, result in zip(playlist_urls, results):
            if isinstance(result, Exception):
                logger.warning("Failed to fetch playlist %s: %s", url, result)
            else:
                manifests.append(result)

        return manifests

    def _fetch_one_manifest(self, playlist_url: str) -> PlaylistManifest:
        """Fetch a single playlist manifest (sync, for thread pool)."""
        ydl_opts = {
            'extract_flat': True,  # Don't download videos, just list them
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)

        if not info:
            raise ValueError(f"Could not extract playlist info: {playlist_url}")

        entries = []
        for i, entry in enumerate(info.get('entries', []) or []):
            if entry is None:
                continue  # Dead/private video
            entries.append(PlaylistEntry(
                video_id=entry.get('id', ''),
                title=entry.get('title'),
                channel=entry.get('uploader') or entry.get('channel'),
                duration=entry.get('duration'),
                upload_date=entry.get('upload_date'),
                playlist_index=i,
            ))

        playlist_id = info.get('id', playlist_url)
        title = info.get('title')

        logger.info("Fetched playlist '%s': %d videos", title or playlist_id, len(entries))

        return PlaylistManifest(
            playlist_id=playlist_id,
            title=title,
            entries=entries,
        )

    # =========================================================================
    # Video processing
    # =========================================================================

    async def _process_videos(
        self,
        video_ids: list[str],
        scorer: ResonanceScorer,
    ) -> list[MemoryShard]:
        """Fetch transcripts and build shards for a batch of videos."""
        shards = []
        yt_spinner = self._get_yt_spinner()

        for video_id in video_ids:
            shard = await self._process_one_video(video_id, yt_spinner, scorer)
            if shard is not None:
                shards.append(shard)

            # Polite rate limiting between transcript fetches
            await asyncio.sleep(0.5)

        return shards

    async def _process_one_video(
        self,
        video_id: str,
        yt_spinner: YouTubeSpinner | None,
        scorer: ResonanceScorer,
    ) -> MemoryShard | None:
        """Process a single video: transcript or fallback to metadata shard."""
        meta = self._resonance_index.get_metadata(video_id)
        title = meta.get('title', 'Unknown')
        channel = meta.get('channel', 'Unknown')
        resonance = self._resonance_index.resonance_count(video_id)

        # Try transcript via YouTubeSpinner
        transcript_text = None
        if yt_spinner is not None:
            try:
                result = await yt_spinner.spin(video_id)
                if result.success and result.shards:
                    transcript_text = result.shards[0].text
            except Exception as e:
                logger.debug("Transcript unavailable for %s: %s", video_id, e)

        # Fallback: title + channel as shard text
        if transcript_text:
            text = transcript_text
            shard_type = 'transcript'
        else:
            text = f"{title} — {channel}"
            shard_type = 'metadata_only'
            logger.debug("Using metadata fallback for %s (%s)", video_id, title)

        # Score importance
        importance = scorer.score(
            text=text,
            video_id=video_id,
            channel=channel,
        )

        # Build shard
        playlists = list(self._resonance_index._sources.get(video_id, set()))

        return self._create_shard(
            id_suffix=video_id,
            text=text,
            episode=f"playlist_scan_{int(time.time())}",
            entities=[channel] if channel and channel != 'Unknown' else [],
            motifs=[],  # Entity/motif extraction happens downstream in the weaving pipeline
            metadata={
                'video_id': video_id,
                'title': title,
                'channel': channel,
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'duration': meta.get('duration'),
                'upload_date': meta.get('upload_date'),
                'shard_type': shard_type,
                'playlists': playlists,
                'resonance_count': resonance,
                'resonance_score': self._resonance_index.resonance_score(video_id),
                'importance_score': importance.score,
                'importance_reason': importance.reason,
            },
        )

    # =========================================================================
    # Checkpoint persistence
    # =========================================================================

    def _load_resonance_checkpoint(self) -> SpinnerCheckpoint:
        """Load checkpoint + resonance index from disk."""
        checkpoint = self.load_checkpoint('playlists')
        if checkpoint is None:
            checkpoint = SpinnerCheckpoint(
                spinner_name=self.name,
                source_id='playlists',
                state={'seen_video_ids': [], 'resonance_index': {}},
            )

        # Restore resonance index
        resonance_data = checkpoint.state.get('resonance_index', {})
        if resonance_data:
            self._resonance_index = ResonanceIndex.from_dict(resonance_data)

        return checkpoint

    def _save_resonance_checkpoint(
        self,
        seen_ids: set[str],
        manifests: list[PlaylistManifest],
    ):
        """Persist checkpoint + resonance index."""
        checkpoint = SpinnerCheckpoint(
            spinner_name=self.name,
            source_id='playlists',
            items_processed=len(seen_ids),
            state={
                'seen_video_ids': list(seen_ids),
                'resonance_index': self._resonance_index.to_dict(),
                'last_manifests': {
                    m.playlist_id: {
                        'title': m.title,
                        'video_count': len(m.entries),
                        'fetched_at': m.fetched_at,
                    }
                    for m in manifests
                },
            },
        )
        self.save_checkpoint(checkpoint)

    # =========================================================================
    # Public helpers
    # =========================================================================

    @property
    def resonance_index(self) -> ResonanceIndex:
        """Access the resonance index for external queries."""
        return self._resonance_index


# =============================================================================
# Convenience Functions
# =============================================================================

async def spin_playlists(
    playlist_urls: list[str],
    batch_size: int = 20,
    checkpoint_dir: str | Path | None = None,
    importance_threshold: float = 0.2,
) -> SpinResult:
    """
    Convenience function to scan YouTube playlists for new content.

    Args:
        playlist_urls: List of YouTube playlist URLs.
        batch_size: Max videos to process per run.
        checkpoint_dir: Where to persist checkpoint state.
        importance_threshold: Min importance to keep a shard.

    Returns:
        SpinResult with transcript shards for new videos.

    Example:
        >>> result = await spin_playlists([
        ...     "https://www.youtube.com/playlist?list=PLxyz",
        ...     "https://www.youtube.com/playlist?list=PLabc",
        ... ])
        >>> for shard in result.shards:
        ...     print(f"{shard.metadata['title']} (resonance: {shard.metadata['resonance_count']})")
    """
    spinner = PlaylistSpinner(
        importance_threshold=importance_threshold,
        batch_size=batch_size,
        checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
    )

    return await spinner.spin(playlist_urls)
